import argparse
import itertools
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import wandb
import yaml
from torch import distributed
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import GPT2Tokenizer

from gpt2 import get_model
from policy import Policy, Reward, DPO
from tldr_dataset import TldrCompletion, TldrPreference
from utils import setup, cleanup, check_fn, fsdp_dict, set_seed, save_model_checkpoint, normalize_reward, fsdp_dict_eval

'''Train models on multiple GPUs using FSDP. '''

# FSDP environment variables and setup
local_rank = int(os.environ['LOCAL_RANK'])  # rank on local node
rank = int(os.environ['RANK'])  # global rank
world_size = int(os.environ['WORLD_SIZE'])  # total number of devices
torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()

def train(config: dict):
    setup()

    set_seed(config['seed'], rank)
    if config['wandb_log'] and rank == 0:  # wandb logging
        wandb_project = 'dpo_grpo_rs'
        wandb_run_name = f"{config['model_for']}-{config['model']}-{str(int(time.time()))}"
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset_dict = {'max_prompt_length': config['max_prompt_length'],
                    'max_response_length': config['max_response_length']}

    # load model
    extra_parameters = {}
    if config['model_for'] == 'sft':
        dataset = TldrCompletion(tokenizer, **dataset_dict)
        model = Policy(FSDP(get_model(config), **fsdp_dict, device_id=device), tokenizer, config, True, device)
    elif config['model_for'] == 'reward':
        sampling_dataset = TldrCompletion(tokenizer, **dataset_dict)
        dataset = TldrPreference(tokenizer, **dataset_dict)
        # policy is for sampling response to normalize reward
        policy_ref = Policy(FSDP(get_model(config), **fsdp_dict_eval, device_id=device), tokenizer, config, False, device)
        policy_ref.lm_model.eval()
        model = Reward(FSDP(get_model(config), **fsdp_dict, device_id=device), tokenizer, config, trained_reward=False, device=device)

        extra_parameters = {'lm_head_gain': model.gain, 'lm_head_bias': model.bias}
        normalize_reward(model, policy_ref, sampling_dataset, config)

    elif config['model_for'] == 'dpo':
        dataset = TldrPreference(tokenizer, **dataset_dict)
        policy_ref = Policy(FSDP(get_model(config), **fsdp_dict, device_id=device), tokenizer, config, False, device)
        policy_ref.lm_model.eval()
        model = DPO(policy_ref, FSDP(get_model(config), **fsdp_dict, device_id=device), tokenizer, config, True, device)
    else:
        raise ValueError(f"Unknown model usage: {config['model_for']}")

    if 'activation_checkpointing' in config and config['activation_checkpointing']:
        apply_activation_checkpointing(model.lm_model, check_fn=check_fn)

    # compile not working.
    # torch._dynamo.exc.InternalTorchDynamoError: RuntimeError: attempting to assign a gradient of size '[32821120]' to a tensor of size '[65642240]'.
    # Please ensure that the gradient and the tensor are the same size
    #model.lm_model = torch.compile(model.lm_model)  # not working.

    model.lm_model.train()

    local_total, local_batch_size = len(dataset) // world_size, config['batch_size'] // world_size
    best_eval_loss = None
    optimizer = model.configure_optimizers(
        itertools.chain(extra_parameters.items(), model.lm_model.named_parameters()), config['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=local_total * config['epoch'] // local_batch_size, eta_min=config['min_lr'])

    def evaluation(current_step: int, force_save:bool = False):
        model.lm_model.eval()
        eval_losses, eval_metrics = [], defaultdict(list)
        local_eval_total = config['eval_sample'] // world_size
        local_eval_batch_size = config['eval_batch_size'] // world_size
        eval_order = rank * dataset.len_val() // world_size + torch.randint(dataset.len_val() // world_size, size=(local_eval_total,))
        for k in range(local_eval_total // local_eval_batch_size):
            vidx = eval_order[k * local_eval_batch_size: k * local_eval_batch_size + local_eval_batch_size]
            eval_input_ids, eval_prompt_length = dataset.get_batch(vidx, train=False)
            eval_input_ids = eval_input_ids.to(device, dtype=torch.int32)
            eval_loss, eval_metric = model.eval_accuracy(eval_input_ids, eval_prompt_length)

            eval_losses.append(eval_loss)
            for mk, mv in eval_metric.items():
                eval_metrics[mk].append(mv)
        model.lm_model.train()

        # sync metrics. change happens in-place
        cur_eval_loss = torch.as_tensor(eval_losses, device=device).mean()
        print(f"eval error rate or loss is {cur_eval_loss.cpu().item():.4f} on rank {rank}")
        distributed.all_reduce(cur_eval_loss, op=distributed.ReduceOp.AVG)
        for mk, mv in eval_metrics.items():
            mv = torch.stack(mv).mean()
            distributed.all_reduce(mv, op=distributed.ReduceOp.AVG)
            eval_metrics[mk] = mv

        nonlocal best_eval_loss
        to_save_dict = {**extra_parameters, 'step': current_step}
        if best_eval_loss is None:
            best_eval_loss = cur_eval_loss
        elif cur_eval_loss < best_eval_loss:
            best_eval_loss = cur_eval_loss
            save_model_checkpoint(model.lm_model, rank, config, to_save_dict)
        elif force_save:
            save_model_checkpoint(model.lm_model, rank, config, to_save_dict)

        if config['wandb_log'] and rank == 0:
            wandb.log({
                f"loss/{config['model_for']}_eval": cur_eval_loss,
                **eval_metrics
            })

    loss_interval = 1
    eval_interval = loss_interval * 100
    for epoch in range(config['epoch']):
        print(f'Start training epoch {epoch}')
        dataset.shuffle(config['seed'] + epoch)  # shuffle before each epoch. use the same seed on all ranks!
        order = rank * local_total + torch.randperm(local_total)
        for j in tqdm(range(local_total // local_batch_size), desc=f"epoch-{epoch} on rank-{rank}"):
            idx = order[j * local_batch_size: j * local_batch_size + local_batch_size]

            # gradient accumulation steps
            micro_batch_size = local_batch_size // config['gradient_step']
            losses, metrics = [], defaultdict(list)
            for j_micro in range(config['gradient_step']):
                slice_idx = slice(j_micro * micro_batch_size, j_micro * micro_batch_size + micro_batch_size)
                input_ids, prompt_length = dataset.get_batch(idx[slice_idx])
                input_ids = input_ids.to(device, dtype=torch.int32)
                loss, metric = model.loss(input_ids, prompt_length)
                (loss / config['gradient_step']).backward()
                # preserve micro-batch metrics
                losses.append(loss.clone().detach())
                for mk, mv in metric.items():
                    metrics[mk].append(mv)

            if config['model_for'] == 'reward':
                distributed.all_reduce(model.gain.grad, op=distributed.ReduceOp.AVG)
                distributed.all_reduce(model.bias.grad, op=distributed.ReduceOp.AVG)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.lm_model.parameters(), float('inf')).detach()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()  # release memory right away

            if j % eval_interval == 0:
                evaluation(epoch * local_total // local_batch_size + j)

            # sync metrics. change happens in-place
            distributed.all_reduce(grad_norm, op=distributed.ReduceOp.AVG)
            fsdp_loss = torch.stack(losses).mean()
            distributed.all_reduce(fsdp_loss, op=distributed.ReduceOp.AVG)
            for mk, mv in metrics.items():
                mv = torch.stack(mv).mean()
                distributed.all_reduce(mv, op=distributed.ReduceOp.AVG)
                metrics[mk] = mv

            if j % loss_interval == 0 and rank == 0:
                if config['wandb_log']:
                    wandb.log({
                        f"loss/{config['model_for']}_train": fsdp_loss,
                        f"lr/{config['model_for']}": scheduler.get_last_lr()[0],
                        f"grad_norm/{config['model_for']}": grad_norm,
                        **metrics
                    })
                else:
                    print(f"training loss is {fsdp_loss.item():.4f}")

    # at training end
    evaluation(config['epoch'] * local_total // local_batch_size, True)

    distributed.barrier()
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_log", action="store_true")
    args = parser.parse_args()

    if not Path(args.config_file).is_file():
        raise ValueError(f"Cannot find configuration file: {args.config_file}")
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # GPU and communication support
    support_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.cuda.nccl.version() >= (2, 10)
    if not support_bf16:
        print("Must install GPUs that support bfloat16.")
        sys.exit(0)

    config['seed'] = args.seed
    config['wandb_log'] = args.wandb_log
    config['parallel'] = 'FSDP'

    train(config)


if __name__ == '__main__':
    main()

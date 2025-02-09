import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch import distributed
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import GPT2Tokenizer

import wandb
from gpt2 import get_model
from policy import Policy, Reward
from tldr_dataset import TldrCompletion, TldrPreference
from utils import setup, cleanup, check_fn, fsdp_dict, set_seed, save_model_checkpoint

'''Train models on multiple GPUs using FSDP. '''


def train(config: dict):
    set_seed(config['seed'])

    # FSDP environment variables and setup
    local_rank = int(os.environ['LOCAL_RANK'])  # rank on local node
    rank = int(os.environ['RANK'])  # global rank
    world_size = int(os.environ['WORLD_SIZE'])  # total number of devices
    setup()
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    if config['wandb_log'] and rank == 0:  # wandb logging
        wandb_project = 'rejection_sampling'
        wandb_run_name = f"{config['model']}-{config['model_for']}-{str(int(time.time()))}"
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # load model
    if config['model_for'] == 'sft':
        dataset = TldrCompletion(tokenizer)
        model = Policy(FSDP(get_model(config), **fsdp_dict, device_id=device),
                        tokenizer, config, True, device)
    elif config['model_for'] == 'reward':
        dataset = TldrPreference(tokenizer)
        model = Reward(FSDP(get_model(config), **fsdp_dict, device_id=device),
                       tokenizer, config, trained_reward=False, device=device)
    else:
        raise ValueError(f"Unknown model usage: {config['model_for']}")

    if 'activation_checkpointing' in config and config['activation_checkpointing']:
        apply_activation_checkpointing(model.lm_model, check_fn=check_fn)
    model.lm_model.train()

    local_total, local_batch_size = len(dataset) // world_size, config['batch_size'] // world_size
    best_eval_loss = None
    optimizer = model.configure_optimizers(config['lr'], wrapped=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=local_total // local_batch_size, eta_min=config['min_lr'])

    local_eval_total = dataset.len_val() // world_size
    eval_order = rank * local_eval_total + torch.arange(local_eval_total)
    def evaluation():
        model.lm_model.eval()
        eval_losses = []
        for k in range(min(50, local_eval_total // local_batch_size)):
            vidx = eval_order[k * local_batch_size: k * local_batch_size + local_batch_size]
            eval_input_ids, eval_mask = dataset.get_batch(vidx, False)
            eval_input_ids, eval_mask = eval_input_ids.to(device, dtype=torch.int32), eval_mask.to(device, dtype=torch.int32)
            with torch.no_grad():
                if config['model_for'] == 'sft':
                    eval_loss = model.loss(eval_input_ids, eval_mask)
                elif config['model_for'] == 'reward':
                    eval_loss = 1 - model.eval_accuracy(eval_input_ids, eval_mask)
            eval_losses.append(eval_loss)
        model.lm_model.train()

        cur_eval_loss = torch.as_tensor(eval_losses, device=device).mean()
        if config['model_for'] == 'sft':
            print(f"eval loss is {cur_eval_loss.cpu().item():.4f} on rank {rank}")
        elif config['model_for'] == 'reward':
            print(f"eval error rate is {cur_eval_loss.cpu().item():.4f} on rank {rank}")

        distributed.all_reduce(cur_eval_loss, op=distributed.ReduceOp.AVG)
        nonlocal best_eval_loss
        if best_eval_loss is None:
            best_eval_loss = cur_eval_loss
        elif cur_eval_loss < best_eval_loss:
            best_eval_loss = cur_eval_loss
            save_model_checkpoint(model.lm_model, rank, config)
        if rank == 0:
            wandb.log({'eval': cur_eval_loss})

    evaluation()  # initial eval loss
    loss_interval = 10
    eval_interval = loss_interval * 50
    for epoch in range(config['epoch']):
        print(f'Start training epoch {epoch}')
        dataset.shuffle()  # shuffle before each epoch
        order = rank * local_total + torch.randperm(local_total)
        for j in tqdm(range(local_total // local_batch_size), desc=f"epoch-{epoch} on rank-{rank}"):
            idx = order[j * local_batch_size: j * local_batch_size + local_batch_size]
            input_ids, mask = dataset.get_batch(idx)
            input_ids, mask = input_ids.to(device, dtype=torch.int32), mask.to(device, dtype=torch.int32)

            loss = model.loss(input_ids, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if j % eval_interval == 0:
                evaluation()

            fsdp_loss = loss.clone().detach()
            distributed.all_reduce(fsdp_loss, op=distributed.ReduceOp.AVG)
            if j % loss_interval == 0 and rank == 0:
                if config['wandb_log']:
                    wandb.log({
                        "loss": fsdp_loss,
                        "lr": scheduler.get_lr()[0]
                    })
                else:
                    print(f"training loss is {fsdp_loss.item():.4f}")
        # at epoch end
        evaluation()

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

    train(config)


if __name__ == '__main__':
    main()

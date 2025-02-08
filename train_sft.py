import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import GPT2Tokenizer

from gpt2 import get_model
from policy import Policy
from tldr_dataset import TldrCompletion
from utils import save_ckpt

'''This script is for testing small models on a single GPU. '''

def train(config: dict):

    seed = 1337 + config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    if config['wandb_log']:  # wandb logging
        wandb_project = 'rejection_sampling'
        wandb_run_name = str(int(time.time()))
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    device = config['device']

    # prepare data
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = TldrCompletion(tokenizer)

    # load model
    policy = Policy(get_model(config), tokenizer, config, True, device)
    policy.lm_model.train()
    policy.lm_model.to(device)

    total, batch_size = len(dataset), config['batch_size']
    best_eval_loss = None
    optimizer = policy.configure_optimizers(config['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=total // batch_size, eta_min=config['min_lr'])

    def eval_loss():
        policy.lm_model.eval()
        eval_losses = []
        eval_order = torch.arange(dataset.len_val())
        for k in range(dataset.len_val() // batch_size):
            vidx = eval_order[k * batch_size: k * batch_size + batch_size]
            eval_input_ids, eval_mask = dataset.get_batch(vidx, False)
            eval_input_ids, eval_mask = eval_input_ids.to(device, dtype=torch.int32), eval_mask.to(device, dtype=torch.int32)
            with torch.no_grad():
                eval_loss = policy.sft_loss(eval_input_ids, eval_mask)
            eval_losses.append(eval_loss.cpu().item())

        cur_eval_loss = np.mean(eval_losses)
        print(f"eval loss is {cur_eval_loss:.4f}")
        policy.lm_model.train()

        nonlocal best_eval_loss
        if best_eval_loss is None:
            best_eval_loss = cur_eval_loss
        elif cur_eval_loss < best_eval_loss:
            best_eval_loss = cur_eval_loss
            save_ckpt(policy.lm_model, config['model'])

    eval_loss()  # initial eval loss
    loss_interval = 100
    eval_interval = loss_interval * 5
    for i in range(config['epoch']):

        print(f'Start training epoch {i}')
        order = torch.randperm(total)
        for j in tqdm(range(total // batch_size)):
            idx = order[j * batch_size: j * batch_size + batch_size]
            input_ids, mask = dataset.get_batch(idx)
            input_ids, mask = input_ids.to(device, dtype=torch.int32), mask.to(device, dtype=torch.int32)

            loss = policy.loss(input_ids, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if config['wandb_log']:
                wandb.log({
                    "loss": loss,
                    "lr": scheduler.get_lr()[0]
                })
            elif j % loss_interval == 0:
                print(f"training loss is {loss.detach().cpu().item():.4f}")

            if j % eval_interval == 0:
                eval_loss()
    eval_loss()


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
    config['device'] = 0  # this script is for single GPU debug

    config['seed'] = args.seed
    config['wandb_log'] = args.wandb_log

    print(config)

    train(config)


if __name__ == '__main__':
    main()

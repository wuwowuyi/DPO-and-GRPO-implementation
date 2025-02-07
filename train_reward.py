import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import GPT2Tokenizer

import wandb
from gpt2 import get_model
from policy import Reward
from tldr_dataset import TldrPreference

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
    dataset = TldrPreference(tokenizer)

    # load model
    reward = Reward(get_model(config), tokenizer, config, trained_reward=False, device=device)
    reward.lm_model.train()
    reward.lm_model.to(device)

    total, batch_size = len(dataset), config['batch_size']
    best_eval_acc = None
    optimizer = reward.configure_optimizers(config['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=total // batch_size, eta_min=config['min_lr'])

    def save_ckpt():
        ckpt = {
            'model': reward.lm_model.state_dict()
        }
        dir = Path('saved_ckpt')
        dir.mkdir(exist_ok=True)
        torch.save(ckpt, dir / f"best_reward_{config['model']}.pt")
        print(f'Best SFT model has been saved')

    def eval_accuracy():
        reward.lm_model.eval()
        eval_acc = []
        eval_order = torch.arange(dataset.len_val())
        for k in range(100):
            vidx = eval_order[k * batch_size: k * batch_size + batch_size]
            eval_input_ids, eval_mask = dataset.get_batch(vidx, False)
            eval_input_ids, eval_mask = eval_input_ids.to(device, dtype=torch.int32), eval_mask.to(device, dtype=torch.int32)
            acc = reward.eval_accuracy(eval_input_ids, eval_mask)
            eval_acc.append(acc.cpu().item())

        cur_eval_acc = np.mean(eval_acc)
        print(f"eval accuracy is {cur_eval_acc:.4f}")
        reward.lm_model.train()

        nonlocal best_eval_acc
        if best_eval_acc is None:
            best_eval_acc = cur_eval_acc
        elif cur_eval_acc > best_eval_acc:
            best_eval_acc = cur_eval_acc
            save_ckpt()

    eval_accuracy()  # initial eval loss
    loss_interval = 1  # TODO
    eval_interval = loss_interval * 5
    for i in range(config['epoch']):

        print(f'Start training epoch {i}')
        order = torch.randperm(total)
        for j in tqdm(range(total // batch_size)):
            idx = order[j * batch_size: j * batch_size + batch_size]
            input_ids, mask = dataset.get_batch(idx)
            input_ids, mask = input_ids.to(device, dtype=torch.int32), mask.to(device, dtype=torch.int32)

            loss = reward.loss(input_ids, mask)
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
                eval_accuracy()
    eval_accuracy()


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

import functools
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, distributed
from torch.distributed.fsdp import MixedPrecision, StateDictType, FullStateDictConfig, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import functional as F


local_rank = int(os.environ['LOCAL_RANK'])
rank = int(os.environ['RANK'])  # global rank
world_size = int(os.environ['WORLD_SIZE'])  # total number of devices
torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()


def setup():
    # initialize the process group
    distributed.init_process_group("nccl")


def cleanup():
    distributed.destroy_process_group()


def check_fn(submodule: nn.Module) -> bool:
    """will be passed each child submodule and returns
            `True` or `False` depending on whether the submodule should be wrapped."""
    return isinstance(submodule, GPT2Block)  # same as wrapping


bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,  # Gradient communication precision.
    buffer_dtype=torch.bfloat16  # Buffer precision.
)
fsdp_dict = {
    'auto_wrap_policy': functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block}),
    'mixed_precision': bfSixteen,
    'use_orig_params': True  # otherwise optimizer configure parameter won't work
}

fsdp_dict_eval = {
    'auto_wrap_policy': functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block}),
    'mixed_precision': bfSixteen,
    #'use_orig_params': True,  # otherwise optimizer configure parameter won't work
    'cpu_offload': CPUOffload(offload_params=True)
}


def set_seed(seed: int, rank: int):
    seed = 100003 * rank + seed  # different seed on each rank
    torch.manual_seed(seed)
    np.random.seed(seed)


def move_padding_left(tokens, pad_token_id):
    """
    Move all padding tokens to the left.
    """
    return torch.tensor(
        [[pad_token_id] * (t == pad_token_id).sum() + [x for x in t if x != pad_token_id] for t in tokens],
        device=tokens.device
    )

def take_top_p_logits(logits: torch.Tensor, p: float):
    """Nucleus sampling.
    The implementation here is to find the minimum logits and use them to filter.
    """
    sorted_logits = torch.sort(logits, descending=True, dim=-1)[0]
    probs = F.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    mask = torch.cumsum(cum_probs >= p, dim=-1) <= 1
    selected = torch.where(mask, sorted_logits, float('inf'))
    min_logits = torch.min(selected, dim=-1, keepdim=True)[0]
    return torch.where(logits >= min_logits, logits, -float('inf'))


def normalize_reward(reward_model, policy, sampling_dataset, config):
    """
    Set reward model gain and bias to get reward mean 0 and std 1.
    """
    @torch.no_grad()
    def sampling_rewards():
        rewards = []
        sample_order = rank * local_total + torch.randint(local_total, size=(sample_local_total,))
        for si in range(sample_local_total // local_batch_size):
            sample_idx = slice(si * local_batch_size, si * local_batch_size + local_batch_size)
            prompt_ids, _ = sampling_dataset.get_batch(sample_order[sample_idx], with_completion=False)
            prompt_ids = prompt_ids.to(device, dtype=torch.int32)
            responses, _ = policy.generate(prompt_ids)
            input_ids = torch.cat((prompt_ids, responses), dim=1)
            rewards.append(reward_model.compute_reward(input_ids))
        return rewards

    sample_total = config['normalize_sample']
    batch_size = config['normalize_batch_size'] if 'normalize_batch_size' in config else config['batch_size']
    sample_local_total, local_batch_size = sample_total // world_size, batch_size // world_size
    local_total = len(sampling_dataset) // world_size

    # compute rewards mean and std to set gain and bias
    rewards = sampling_rewards()
    all_rewards = torch.zeros(sample_total, dtype=rewards[0].dtype, device=device)
    distributed.all_gather_into_tensor(all_rewards, torch.cat(rewards))
    gain, bias = reward_model.compute_gain_bias(all_rewards.mean(), all_rewards.std())
    reward_model.set_gain_bias(gain, bias)

    # validate mean and std are now close to 0 and 1. if not, increase normalize_sample.
    rewards = sampling_rewards()
    distributed.all_gather_into_tensor(all_rewards, torch.cat(rewards))
    print(f"after normalization, the mean and std {all_rewards.mean():.4f}, {all_rewards.std():.4f}")


model_ckpt_dir = Path('saved_ckpt')
model_ckpt_dir.mkdir(parents=True, exist_ok=True)

def save_ckpt(model: nn.Module, model_save_name: str):
    """
    Used by single GPU training
    """
    ckpt = {
        'model': model.state_dict()
    }
    torch.save(ckpt, model_ckpt_dir / f"best_{model_save_name}.pt")
    print(f'Best SFT model has been saved')


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

def save_model_checkpoint(model, rank, cfg, extra_state_dict=None):
    """saving model via rank0 cpu streaming and full_state_dict"""

    # saving with rank0 cpu
    if not cfg['checkpoint_type'] == 'FULL_STATE_DICT':
        print(f" unable to handle checkpoint type {cfg['checkpoint_type']}, aborting")

    if extra_state_dict is None:
        extra_state_dict = {}
    step = extra_state_dict.pop('step', '')

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
        ckpt = {
            'model': model.state_dict(),
            **extra_state_dict
        }
    if rank == 0:
        save_name = f"{cfg['model_for']}_{cfg['model']}_fsdp_{step}_{str(int(time.time()))}.pt"
        print(f"--> saving model {save_name} ...")
        torch.save(ckpt, model_ckpt_dir / save_name)


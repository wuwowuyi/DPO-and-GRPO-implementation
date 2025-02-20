import functools
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, distributed
from torch.distributed.fsdp import MixedPrecision, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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

def save_model_checkpoint(model, rank, cfg, extra_state_dict):
    """saving model via rank0 cpu streaming and full_state_dict"""

    # saving with rank0 cpu
    if not cfg['checkpoint_type'] == 'FULL_STATE_DICT':
        print(f" unable to handle checkpoint type {cfg['checkpoint_type']}, aborting")

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
        ckpt = {
            'model': model.state_dict(),
            **extra_state_dict
        }
    if rank == 0:
        save_name = f"{cfg['model_for']}_{cfg['model']}_fsdp_{str(int(time.time()))}.pt"
        print(f"--> saving model {save_name} ...")
        torch.save(ckpt, model_ckpt_dir / save_name)


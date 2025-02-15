from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Config


model_cache = 'cached_model'
Path(model_cache).mkdir(parents=True, exist_ok=True)

@dataclass
class GPT2ModelParams:
    n_positions: int = 1024  # max_sequence_length

    # dropout probs
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

@dataclass
class GPT2LargeModelParams(GPT2ModelParams):
    n_layer: int = 36
    n_head: int = 20
    n_embd: int = 1280

@dataclass
class GPT2XLargeModelParams(GPT2ModelParams):
    n_layer: int = 48
    n_head: int = 25
    n_embd: int = 1600


model_param = {
    'gpt2': GPT2ModelParams,
    'gpt2-large': GPT2LargeModelParams,
    'gpt2-xl': GPT2XLargeModelParams
}


def get_model(config: dict) -> GPT2LMHeadModel:
    """
    Load a GPT-2 model from checkpoint, or hugging face pretrained.
    """
    model_type = config['model']
    mcfg = GPT2Config(**asdict(model_param[model_type]()))

    if 'ckpt' in config:
        if not Path(config['ckpt']).is_file():
            raise ValueError(f"checkpoint does not exist: {config['ckpt']}")
        print(f"Load model {model_type} from checkpoint {config['ckpt']}")
        ckpt = torch.load(config['ckpt'], weights_only=True)
        model = GPT2LMHeadModel(mcfg)
        if config['ckpt_type'] == 'reward':
            model.lm_head = nn.Linear(mcfg.n_embd, 1, bias=False)
        model.load_state_dict(ckpt.pop('model'))
        #model.to(dtype=torch.bfloat16)  # seems to brings down performance
    else:
        print(f"Load a pretrained model {model_type} from Hugging Face pretrained")
        model = GPT2LMHeadModel.from_pretrained(model_type, config=mcfg, cache_dir=model_cache, torch_dtype=torch.float32)

    return model


if __name__ == '__main__':
    config = {'model': 'gpt2', 'ckpt': 'model_124M/124M_ckpt.pt'}
    print(get_model(config))

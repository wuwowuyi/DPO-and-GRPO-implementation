import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer

from gpt2 import GPT2ModelParams, model_param


class Policy:
    def __init__(
            self,
            lm_model,  # language model
            tokenizer: GPT2Tokenizer,
            config: dict,
            train: bool = True,
            device: int = 0  # index of current device
    ):
        super().__init__()
        self.lm_model = lm_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        if not train:
            for param in self.lm_model.parameters():
                param.requires_grad_(False)

    def sft_loss(self, input_ids, mask):
        """
        Supervised fine-tuning loss.

        :param input_ids: concatenation of prompt + completion token. shape=(batch_size, sequence_length)
        :param mask: where 0 corresponds the pad token in input_ids
        :return:
        """
        input_token, attention_mask = input_ids[:, :-1].clone(), mask[:, :-1]
        label = input_ids[:, 1:]
        label_mask = torch.ne(mask[:, 1:], 0).to(device=self.device)

        logits = self.lm_model(input_token, attention_mask=attention_mask).logits  # shape=(batch_size, sequence_length-1, vocab_size)
        b, t = label.shape
        loss = F.cross_entropy(logits.reshape(b * t, -1), label.reshape(-1).long(), reduction='none')
        loss = torch.mean(loss.reshape(b, t) * label_mask)  # ignore loss on the padding tokens
        return loss

    def configure_optimizers(self, learning_rate, weight_decay=1e-2):
        """
        Adapted from nanoGPT https://github.com/karpathy/nanoGPT
        """
        def prepare_arguments():
            # start with all of the candidate parameters
            param_dict = {pn: p for pn, p in self.lm_model.named_parameters()}
            # filter out those that do not require grad
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            return optim_groups

        optim_groups = prepare_arguments()
        optimizer = torch.optim.AdamW(optim_groups, learning_rate, betas=(0.9, 0.95))
        return optimizer


class Reward(Policy):
    def __init__(
            self,
            lm_model,  # language model
            tokenizer: GPT2Tokenizer,
            config: dict,
            trained_reward: bool = False,  # is the reward model trained
            device: int = 0  # index of current device
    ):
        super().__init__(lm_model, tokenizer, config, not trained_reward, device)

        if not trained_reward:
            n_embd = model_param[config['model']].n_embd  # dimension of embedding, i.e. hidden_size
            self.lm_model.lm_head = nn.Linear(n_embd, 1)
            torch.nn.init.normal_(self.lm_model.lm_head.weight, std=1 / np.sqrt(n_embd + 1))

    def forward(self, input_ids, mask):
        logits = self.lm_model(input_ids, attention_mask=mask).logits
        logits = logits.squeeze()[:, -1]  # shape=(batch_size * 2,)
        rewards = torch.stack(torch.tensor_split(logits, 2)).T  # shape=(batch_size, 2)
        return rewards

    def loss(self, input_ids, mask):
        rewards = self.forward(input_ids, mask)
        labels = torch.zeros(len(input_ids) // 2).to(self.device, dtype=torch.int64)  # shape=(batch_size,)
        loss = F.cross_entropy(rewards, labels)
        return loss

    @torch.no_grad()
    def eval_accuracy(self, input_ids, mask):
        rewards = self.forward(input_ids, mask)
        acc = torch.mean(rewards[:, 0] > rewards[:, 1], dtype=torch.float16)
        return acc

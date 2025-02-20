import contextlib
from operator import itemgetter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from gpt2 import model_param


class LLM:
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

            self.lm_model.to(dtype=torch.bfloat16)

    def loss(self, input_ids: torch.Tensor, mask: torch.Tensor, prompt_length: int, **kwargs):
        """
        Supervised fine-tuning loss.

        :param input_ids: concatenation of prompt + completion token. shape=(batch_size, sequence_length)
        :param mask: mask of a padding token should be zero otherwise 1.
        :param prompt_length: length of prompt in sequence
        :return:
        """
        input_token, attention_mask = input_ids[:, :-1].clone(), mask[:, :-1]
        logits = self.lm_model(input_token, attention_mask=attention_mask).logits[:, prompt_length-1:, :]  # shape=(n, response_length, vocab_size)
        label = input_ids[:, prompt_length:]  # we only care the response
        b, t = label.shape
        loss = F.cross_entropy(logits.reshape(b * t, -1), label.reshape(-1).long(), reduction='none')
        loss = torch.mean(loss.reshape(b, t) * mask[:, prompt_length:])  # ignore loss on the padding tokens
        return loss, {}

    @torch.no_grad()
    def eval_accuracy(self, input_ids, mask, prompt_length):
        return self.loss(input_ids, mask, prompt_length)

    @staticmethod
    def configure_optimizers(parameters, learning_rate, weight_decay=1e-2):
        """
        Adapted from nanoGPT https://github.com/karpathy/nanoGPT
        """
        def prepare_arguments():
            # start with all of the candidate parameters
            param_dict = {pn: p for pn, p in parameters}
            # filter out those that do not require grad
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params, nodecay_params = [], []
            for n, p in param_dict.items():
                if 'ln_' in n or 'bias' in n or 'gain' in n:
                    nodecay_params.append(p)
                else:
                    decay_params.append(p)
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


class Policy(LLM):

    def log_prob(self, input_ids, mask, prompt_length: int, avg_seq: bool = True, scale: bool = False):
        """
        Compute per token(step) log probability.

        :param input_ids: shape=(n, sequence_length)
        :param mask: shape=(n, sequence_length). mask of a padding token should be zero otherwise 1.
        :param prompt_length: length of prompt in sequence
        :param avg_seq: whether to take an average over the entire response

        :return: logp shape=(n,) if average over response else shape=(n, response_length)
        """
        input_token, attention_mask = input_ids[:, :-1].clone(), mask[:, :-1]
        logits = self.lm_model(input_token, attention_mask=attention_mask).logits[:, prompt_length-1:, :]  # shape=(n, response_length, vocab_size)

        if scale:  # scaled by temperature
            logits /= torch.as_tensor(self.config['temperature'], dtype=torch.bfloat16, device=self.device)

        label = input_ids[:, prompt_length:]  # shape=(n, response_length)
        n, response_length = label.shape
        logp = -F.cross_entropy(logits.reshape(n * response_length, -1), label.reshape(-1).long(), reduction='none')  # shape=(n * response_length,)
        logp = logp.reshape(n, response_length) * mask[:, prompt_length:]  # ignore logp on the padding tokens
        return torch.sum(logp, dim=-1) if avg_seq else logp

    @torch.no_grad()
    def generate(self, prompt, mask, n_samples: int =1, max_response_length: int = 64, return_logp: bool = False):
        """
        Generate responses given prompt.
        The generated samples of the same prompt are consecutive in the responses.
        Let's say n_samples = 8, then
        responses[:8] are responses to the first prompt, responses[8:16] second prompt, and so on.

        NOTE:
        1) prompt padding tokens must be on the left!
        2) do NOT use position_ids

        :param prompt: shape=(n, prompt_length)
        :param mask: shape=(n, prompt_length). mask of a padding token should be zero otherwise 1.
        :param n_samples: number of responses for each prompt
        :param max_response_length: max length of generated response
        :param return_logp: log prob of generated response

        :return: sampled responses and optionally corresponding log probs.
        """
        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.lm_model, recurse=False, writeback=False)
                       if self.config['parallel'] == 'FSDP' else contextlib.nullcontext())
        with ctx():
            outputs = self.lm_model.generate(
                prompt,
                attention_mask=mask,
                max_new_tokens=max_response_length,
                #position_ids=torch.cumsum(mask, dim=1) - mask, # do NOT work. very bad response
                do_sample=True,  # random sampling
                temperature=self.config['temperature'],
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=n_samples,
                return_dict_in_generate=True,
                output_logits=True
            )
        _, prompt_length = prompt.shape
        responses = outputs.sequences[:, prompt_length:]  # shape=(n * n_samples, response_length)
        logps = None
        if return_logp:
            logits = torch.stack(outputs.logits)  # shape=(response_length, n * n_samples, vocab_size)
            logps = F.log_softmax(logits.transpose(0, 1), dim=-1, dtype=torch.bfloat16)  # shape=(n * n_samples, response_length, vocab_size)
            logps = torch.gather(logps, -1, responses.unsqueeze(2)).squeeze(2)  # shape=(n * n_samples, response_length)
        return responses.to(dtype=torch.int32, device=self.device), logps.to(device=self.device) if logps else logps


class Reward(LLM):
    """
    For all methods, if input_ids are preference pairs, first half are prompt + chosen, self half prompt + rejected.
    All padding tokens of a prompt are on the LEFT side!

    input_ids.shape=(n, sequence_length)
    mask.shape=(n, sequence_length). mask of a padding token should be zero otherwise 1.
    """
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
            if isinstance(lm_model, FSDP):
                self.lm_model.module.lm_head = nn.Linear(n_embd, 1, bias=False, device=self.device, dtype=torch.bfloat16)
                torch.nn.init.normal_(self.lm_model.module.lm_head.weight, std=1 / np.sqrt(n_embd + 1))
            else:
                self.lm_model.lm_head = nn.Linear(n_embd, 1, bias=False, device=self.device, dtype=torch.bfloat16)
                torch.nn.init.normal_(self.lm_model.lm_head.weight, std=1 / np.sqrt(n_embd + 1))

            self.gain = torch.nn.Parameter(torch.tensor(1.0, device=self.device, dtype=torch.bfloat16), requires_grad=True)
            self.bias = torch.nn.Parameter(torch.tensor(0.0, device=self.device, dtype=torch.bfloat16), requires_grad=True)

    @torch.no_grad()
    def set_gain_bias(self, mean, std) -> None:
        """
        Compute and set gain and bias given current mean ans std, to get target mean 0 and std 1.

        :param mean: current mean
        :param std: current std
        """
        print(f"current mean {mean} and std {std}")
        target_mean = torch.tensor(0.0, dtype=mean.dtype, device=self.device)
        target_std = torch.tensor(1.0, dtype=std.dtype, device=self.device)
        gain = target_std / std
        bias = target_mean - gain * mean
        print(f"set gain and bias to {gain:.4f} and {bias:.4f}")
        self.gain.copy_(gain)
        self.bias.copy_(bias)

    def compute_reward(self, input_ids, mask):
        """
        input_ids = prompt + response, to compute reward of the entire sequence

        Note: ALL paddings tokens are on the left!

        :param input_ids: shape=(n, sequence_length)
        :param mask: shape=(n, sequence_length)
        :return: rewards. shape=(n,)
        """
        # important for GPT2. ALL padding on left otherwise this is wrong.
        # All the padding tokens have position id 0, and the sequence is 0, 1, ...
        position_ids = torch.cumsum(mask, dim=1) - mask

        output = self.lm_model(input_ids, attention_mask=mask, position_ids=position_ids)  # shape=(n, sequence_length, 1)
        rewards = output.logits.squeeze(2)[:, -1]
        return rewards * self.gain + self.bias.expand_as(rewards)

    def _compute_loss(self, rewards):
        rewards = torch.stack(torch.tensor_split(rewards, 2)).transpose(0, 1)  # shape=(n//2, 2)
        labels = torch.zeros(len(rewards)).to(self.device, dtype=torch.int64)  # first half chosen
        loss = F.cross_entropy(rewards, labels)
        return loss

    def loss(self, input_ids, mask, prompt_length, **kwargs):
        rewards = self.compute_reward(input_ids, mask)  # shape=(n, )
        loss = self._compute_loss(rewards)
        k = rewards.shape[0] // 2
        return loss, {
            f"reward/{self.config['model_for']}_train_chosen": torch.mean(rewards[:k]),
            f"reward/{self.config['model_for']}_train_rejected": torch.mean(rewards[k:]),
            f"reward/{self.config['model_for']}_train_acc": (rewards[:k] > rewards[k:]).float().mean(),
            f"reward/{self.config['model_for']}_gain": self.gain,
            f"reward/{self.config['model_for']}_bias": self.bias,
        }

    @torch.no_grad()
    def eval_accuracy(self, input_ids, mask, prompt_length):
        rewards = self.compute_reward(input_ids, mask)  # shape=(n, )
        loss = self._compute_loss(rewards)
        k = rewards.shape[0] // 2
        return loss, {
            f"reward/{self.config['model_for']}_eval_chosen": torch.mean(rewards[:k]),
            f"reward/{self.config['model_for']}_eval_rejected": torch.mean(rewards[k:]),
            f"reward/{self.config['model_for']}_eval_acc": (rewards[:k] > rewards[k:]).float().mean()
        }


class DPO(Policy):
    """
    For all methods, if input_ids are preference pairs, first half are prompt + chosen, self half prompt + rejected.
    All padding tokens of a prompt are on the LEFT side!

    input_ids.shape=(n, sequence_length)
    mask.shape=(n, sequence_length). mask of a padding token should be zero otherwise 1.
    """

    def __init__(
            self,
            policy_ref: Policy,
            lm_model,  # language model
            tokenizer: GPT2Tokenizer,
            config: dict,
            train: bool = True,
            device: int = 0  # index of current device
    ):
        super().__init__(lm_model, tokenizer, config, train, device)
        self.policy_ref = policy_ref

    def _compute_loss_reward(self, logp, logp_ref):
        b = logp.shape[0] // 2
        # win_idx: index of prompt + chosen response.
        # lose_idx: index of prompt + rejected response.
        win_idx, lose_idx = torch.arange(b, device=self.device), torch.arange(b, 2 * b, device=self.device)
        logp_ref_w, logp_ref_l = logp_ref[win_idx], logp_ref[lose_idx]
        logp_w, logp_l = logp[win_idx], logp[lose_idx]
        loss_logits = (logp_w - logp_l) - (logp_ref_w - logp_ref_l)
        beta, label_smoothing = self.config['beta'], self.config['label_smoothing']
        loss = -F.logsigmoid(beta * loss_logits) * (1 - label_smoothing) - F.logsigmoid(-beta * loss_logits) * label_smoothing
        reward_w = beta * (logp_w.detach() - logp_ref_w)
        reward_l = beta * (logp_l.detach() - logp_ref_l)
        return loss.mean(), reward_w, reward_l

    def loss(self, input_ids, mask, prompt_length, **kwargs):
        with torch.no_grad():
            logp_ref = self.policy_ref.log_prob(input_ids, mask, prompt_length)
        logp = self.log_prob(input_ids, mask, prompt_length)

        loss, reward_w, reward_l = self._compute_loss_reward(logp, logp_ref)
        return loss, {
            f"reward/{self.config['model_for']}_train_chosen": reward_w.mean(),
            f"reward/{self.config['model_for']}_train_rejected": reward_l.mean(),
            f"reward/{self.config['model_for']}_train_acc": (reward_w > reward_l).float().mean()
        }

    @torch.no_grad()
    def eval_accuracy(self, input_ids, mask, prompt_length):
        logp_ref = self.policy_ref.log_prob(input_ids, mask, prompt_length)
        logp = self.log_prob(input_ids, mask, prompt_length)

        loss, reward_w, reward_l = self._compute_loss_reward(logp, logp_ref)
        return loss, {
            f"reward/{self.config['model_for']}_eval_chosen": reward_w.mean(),
            f"reward/{self.config['model_for']}_eval_rejected": reward_l.mean(),
            f"reward/{self.config['model_for']}_eval_acc": (reward_w > reward_l).float().mean()
        }


class RejectionSampling:
    """
    Rejection sampling. see paper https://arxiv.org/abs/2309.06657.
    """
    def _post_process(self, responses):
        """applied to responses before computing a reward.
        Set all tokens after the truncate token to padding token.

        responses.shape=(n, response_length)
        """
        masks = []
        for t in self.truncate_tokens:
            masks.append(torch.eq(responses, t).int())  # like 0, 0, ..., 1, 0, ...0, 1, 0,..
        mask = torch.maximum(*masks)  # set any truncate token match to 1
        mask = torch.cumsum(mask, dim=1) - mask  # all 0 before and at the first truncate token, all 1 after.
        processed = torch.where(mask.bool(), self.tokenizer.pad_token_id, responses)
        processed_mask = torch.eq(mask, 0)
        return processed, processed_mask

    def _filter_response(self, responses):
        masks = []
        for t in self.truncate_tokens:
            masks.append(torch.eq(responses, t).int())  # like 0, 0, ..., 1, 0, ...0, 1, 0,..
        mask = torch.maximum(*masks)
        return torch.any(mask, dim=1)

    @torch.no_grad()
    def compute_rewards(self, reward_model: Reward, prompt, mask, responses, n_samples: int, penalty_reward_value: float):
        # post process responses
        processed, processed_mask = self._post_process(responses)
        # call reward_model to compute rewards
        reward_input = torch.cat((prompt, processed), dim=1)  # prompt left padded, response right padded.
        reward_mask = torch.cat((mask, processed_mask), dim=1)
        raw_rewards = reward_model.compute_reward(reward_input, reward_mask)  # shape=(n * n_samples,)
        # normalize rewards per group
        rewards = torch.stack(torch.split(raw_rewards, n_samples))  # shape=(n, n_samples)
        group_mean, group_std = rewards.mean(dim=1, keepdim=True), rewards.std(dim=1, keepdim=True)
        rewards = ((rewards - group_mean) / group_std).reshape(-1)  # shape=(n * n_samples,)
        # penalize reward
        valid_mask = self._filter_response(responses)  # shape=(n * n_samples,)
        rewards = torch.where(valid_mask, rewards, penalty_reward_value)  # shape=(n * n_samples,)
        return rewards

    def rejection_sampling(self, responses, rewards, num_samples, beta):
        """
        Algorithm 1 (conduct_rejection_sampling function) from paper https://arxiv.org/abs/2309.06657.

        responses: responses to a prompt
        rewards: rewards of these responses, in the same order as responses
        num_samples: number to select
        beta: beta parameter in KL-constrained reward maximization objective

        return accepted samples, sorted by reward in descending order
        """
        candidates = {c: r for c, r in zip(torch.unbind(responses), torch.unbind(rewards))}
        accepted = []
        while len(accepted) < num_samples:
            max_reward = max(candidates.values())
            to_remove = []
            for c, r in candidates.items():
                u = torch.rand(1, dtype=torch.bfloat16, device=self.device)
                if u >= torch.exp((r - max_reward) / beta):  # todo: range of right side value?
                    continue
                accepted.append((c, r))
                to_remove.append(c)
                if len(accepted) == num_samples:
                    break
            for c in to_remove:
                candidates.pop(c)

        accepted.sort(key=itemgetter(1), reverse=True)  # sort by reward in descending order
        return [t for t, r in accepted]


class RejectionSamplingDPO(DPO, RejectionSampling):
    """
    Rejection sampling + DPO.
    Specifically, use rejection sampling to generate preference pairs to train DPO.
    see paper https://arxiv.org/abs/2309.06657.
    """
    def __init__(
            self,
            reward_model: Reward,  # compute rewards for generated samples
            policy_ref: Policy,  # to generate samples as proposal distribution
            lm_model,  # language model
            tokenizer: GPT2Tokenizer,
            config: dict,
            train: bool = True,
            device: int = 0  # index of current device
    ):
        super().__init__(policy_ref, lm_model, tokenizer, config, train, device)
        self.reward_model = reward_model
        self.truncate_tokens = [198, self.tokenizer.eos_token_id]  # GPT2 tokenizer.encode('\n') = 198

    @torch.no_grad()
    def generate_pairs(self, prompt, mask):
        n_samples, max_response_length = self.config['n_samples'], self.config['max_response_length']
        n, prompt_length = prompt.shape

        # first generate samples using the proposal policy
        responses, _ = self.policy_ref.generate(prompt, mask, n_samples, max_response_length)

        # second, compute rewards of the generated samples
        input_ids = prompt.tile((1, n_samples)).reshape(n * n_samples, -1)  # shape=(n * n_samples, prompt_length)
        tiled_mask = mask.tile((1, n_samples)).reshape(n * n_samples, -1)
        rewards = self.compute_rewards(self.reward_model, input_ids, tiled_mask,
                                       responses, n_samples, self.config['penalty_reward_value'])

        # third, construct pairs for DPO
        seq = torch.cat((input_ids, responses), dim=1)
        t_groups, r_groups = torch.split(seq, n_samples), torch.split(rewards, n_samples)
        chosen, rejected = [], []
        n_select, k = self.config['select_samples'], self.config['select_samples'] // 2
        for t, r in zip(t_groups, r_groups):
            samples = self.rejection_sampling(t, r, n_select, self.config['sampling_beta'])
            chosen.extend(samples[:k])
            rejected.extend(samples[k:])

        input_ids = torch.cat((torch.stack(chosen), torch.stack(rejected)))  # (n * n_select, prompt_length + response_length)
        tiled_mask = mask.tile((1, k)).reshape(n * k, -1)
        tiled_mask = torch.cat((tiled_mask, tiled_mask))
        response_mask = torch.eq(input_ids[:, prompt_length:], self.tokenizer.pad_token_id).int()  # (0, 0, ..., 0, 1, ... 1, 1)
        response_mask = torch.cumsum(response_mask, 1) - response_mask  # first padding token is 0, otherwise 1
        response_mask = torch.eq(response_mask, 0).int()
        input_mask = torch.cat((tiled_mask, response_mask), dim=1)

        return input_ids, input_mask

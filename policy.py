import contextlib
from operator import itemgetter

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import functional as F
from transformers import GPT2Tokenizer

from gpt2 import model_param
from utils import move_padding_left, take_top_p_logits


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

    def loss(self, input_ids: torch.Tensor, prompt_length: int, **kwargs):
        """
        Supervised fine-tuning loss.

        input_ids is a concatenation of prompt + completion tokens.
        Prompt is left padded, and completion right padded.

        :param input_ids:  shape=(batch_size, sequence_length)
        :param prompt_length: length of prompt in sequence
        :return:
        """
        mask = (input_ids != self.tokenizer.pad_token_id).int()
        position_ids = (torch.cumsum(mask, 1) - mask)[:, :-1]
        input_token, attention_mask = input_ids[:, :-1].clone(), mask[:, :-1]

        logits = self.lm_model(input_token, attention_mask=attention_mask, position_ids=position_ids).logits[:, prompt_length-1:, :]  # shape=(n, response_length, vocab_size)
        label = input_ids[:, prompt_length:]  # we only care the response
        b, t = label.shape
        loss = F.cross_entropy(logits.reshape(b * t, -1), label.reshape(-1).long(), reduction='none')
        loss = torch.mean(loss.reshape(b, t) * mask[:, prompt_length:])  # ignore loss on the padding tokens
        return loss, {}

    @torch.no_grad()
    def eval_accuracy(self, input_ids, prompt_length):
        return self.loss(input_ids, prompt_length)

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

    def log_prob(self, input_ids, prompt_length: int, avg_seq: bool = True, scale: bool = False, return_entropy=False, top_p=1.0):
        """
        Compute per token(step) log probability.

        :param input_ids: shape=(n, sequence_length)
        :param prompt_length: length of prompt in sequence
        :param avg_seq: whether to take an average over the entire response

        :return: logp shape=(n,) if average over response else shape=(n, response_length)
        """
        mask = (input_ids != self.tokenizer.pad_token_id).int()
        position_ids = (torch.cumsum(mask, 1) - mask)[:, :-1]
        label, input_ids, attention_mask = input_ids[:, prompt_length:], input_ids[:, :-1], mask[:, :-1]
        logits = self.lm_model(input_ids, attention_mask=attention_mask, position_ids=position_ids).logits[:, prompt_length-1:, :]  # shape=(n, response_length, vocab_size)

        if scale:  # scaled by temperature
            logits /= torch.as_tensor(self.config['temperature'], dtype=torch.bfloat16, device=self.device)

        if top_p < 1.0:
            logits = take_top_p_logits(logits, top_p)

        n, response_length = label.shape
        logp = -F.cross_entropy(logits.reshape(n * response_length, -1), label.reshape(-1).long(), reduction='none')  # shape=(n * response_length,)
        logp = logp.reshape(n, response_length) * mask[:, prompt_length:]  # ignore logp on the padding tokens

        entropy = None
        if return_entropy:
            p = F.softmax(logits, dim=-1)  # shape=(n, response_length, vocab_size)
            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(p * logits, dim=-1)  # shape=(n, response_length)
            entropy = entropy * mask[:, prompt_length:]  # ignore padding tokens entropies

        return torch.sum(logp, dim=-1) if avg_seq else logp, entropy

    def compute_loss_reward(self, logp, logp_ref):
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

    @torch.no_grad()
    def generate(self, prompt, n_samples: int =1, max_response_length: int = 64, return_logp: bool = False):
        """
        Generate responses given prompt.
        The generated samples of the same prompt are consecutive in the responses.
        Let's say n_samples = 8, then
        responses[:8] are responses to the first prompt, responses[8:16] second prompt, and so on.

        NOTE:
        1) prompt padding tokens must be on the left!
        2) do NOT use position_ids

        :param prompt: shape=(n, prompt_length)
        :param n_samples: number of responses for each prompt
        :param max_response_length: max length of generated response
        :param return_logp: log prob of generated response

        :return: sampled responses and optionally corresponding log probs.
        """
        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.lm_model, recurse=False, writeback=False)
                       if self.config['parallel'] == 'FSDP' else contextlib.nullcontext())

        extra_params = {'top_p': 1.0 if 'top_p' not in self.config else self.config['top_p'],
                        'top_k': 0 if 'top_k' not in self.config else self.config['top_k']}

        with ctx(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.lm_model.generate(
                prompt,
                attention_mask=(prompt != self.tokenizer.pad_token_id).int(),
                max_new_tokens=max_response_length,
                #position_ids=torch.cumsum(mask, dim=1) - mask, # do NOT work. very bad response
                do_sample=True,  # random sampling
                temperature=self.config['temperature'],
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=n_samples,
                return_dict_in_generate=True,
                output_logits=True,
                **extra_params
            )
        _, prompt_length = prompt.shape
        responses = outputs.sequences[:, prompt_length:]  # shape=(n * n_samples, response_length)

        logps = None
        if return_logp:
            logits = torch.stack(outputs.logits)  # shape=(response_length, n * n_samples, vocab_size)
            logits = logits.transpose(0, 1)  # shape=(n * n_samples, response_length, vocab_size)
            logits /= torch.as_tensor(self.config['temperature'], dtype=torch.bfloat16, device=self.device)
            if extra_params['top_k'] > 0:
                v, _ = torch.topk(logits, min(extra_params['top_k'], logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')  # use v[:, [-1]] rather than v[:, -1] to keep dims.
            if extra_params['top_p'] < 1.0:
                logits = take_top_p_logits(logits, extra_params['top_p'])

            logps = -F.cross_entropy(logits.flatten(0, 1), responses.reshape(-1).long(), reduction='none')  # shape=(n * n_samples * response_length,)
            logps = logps.reshape(responses.shape)
        return responses.to(dtype=torch.int32), logps


class Reward(LLM):
    """
    For all methods, if input_ids are preference pairs, first half are prompt + chosen, self half prompt + rejected.
    All padding tokens of a prompt are on the LEFT side!

    input_ids.shape=(n, sequence_length)
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

    def eval(self):
        self.lm_model.eval()
        self.gain.requires_grad_(False)
        self.bias.requires_grad_(False)

    @torch.no_grad()
    def compute_gain_bias(self, mean, std):
        """
        Compute gain and bias given current mean and std, to get target mean 0 and std 1.

        :param mean: current mean
        :param std: current std
        """
        print(f"current mean {mean} and std {std}")
        target_mean = torch.tensor(0.0, dtype=mean.dtype, device=self.device)
        target_std = torch.tensor(1.0, dtype=std.dtype, device=self.device)
        gain = target_std / std
        bias = target_mean - gain * mean
        return gain, bias

    @torch.no_grad()
    def set_gain_bias(self, gain, bias):
        print(f"set gain and bias to {gain:.4f} and {bias:.4f}")
        self.gain.copy_(gain)
        self.bias.copy_(bias)

    def compute_reward(self, input_ids):
        """
        input_ids = prompt + response, to compute reward of the entire sequence

        :param input_ids: shape=(n, sequence_length)
        :return: rewards. shape=(n,)
        """
        # important for GPT2. ALL padding on left otherwise this is wrong.
        # All the padding tokens have position id 0, and the rest is 0, 1, ...
        input_ids = move_padding_left(input_ids, self.tokenizer.pad_token_id)
        mask = (input_ids != self.tokenizer.pad_token_id).int()
        position_ids = torch.cumsum(mask, dim=1) - mask

        output = self.lm_model(input_ids, attention_mask=mask, position_ids=position_ids)  # shape=(n, sequence_length, 1)
        rewards = output.logits.squeeze(2)[:, -1]
        return rewards * self.gain + self.bias.expand_as(rewards)

    def _compute_loss(self, rewards):
        rewards = torch.stack(torch.tensor_split(rewards, 2)).transpose(0, 1)  # shape=(n//2, 2)
        labels = torch.zeros(len(rewards)).to(self.device, dtype=torch.int64)  # first half chosen
        loss = F.cross_entropy(rewards, labels)
        return loss

    def loss(self, input_ids, prompt_length, **kwargs):
        rewards = self.compute_reward(input_ids)  # shape=(n, )
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
    def eval_accuracy(self, input_ids, prompt_length):
        rewards = self.compute_reward(input_ids)  # shape=(n, )
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

    def loss(self, input_ids, prompt_length, **kwargs):
        with torch.no_grad():
            logp_ref, _ = self.policy_ref.log_prob(input_ids, prompt_length)
        logp, _ = self.log_prob(input_ids, prompt_length)

        loss, reward_w, reward_l = self.compute_loss_reward(logp, logp_ref)
        return loss, {
            f"reward/{self.config['model_for']}_train_chosen": reward_w.mean(),
            f"reward/{self.config['model_for']}_train_rejected": reward_l.mean(),
            f"reward/{self.config['model_for']}_train_acc": (reward_w > reward_l).float().mean()
        }

    @torch.no_grad()
    def eval_accuracy(self, input_ids, prompt_length):
        logp_ref, _ = self.policy_ref.log_prob(input_ids, prompt_length)
        logp, _ = self.log_prob(input_ids, prompt_length)

        loss, reward_w, reward_l = self.compute_loss_reward(logp, logp_ref)
        return loss, {
            f"reward/{self.config['model_for']}_eval_chosen": reward_w.mean(),
            f"reward/{self.config['model_for']}_eval_rejected": reward_l.mean(),
            f"reward/{self.config['model_for']}_eval_acc": (reward_w > reward_l).float().mean()
        }


class GRPO(Policy):
    """
    GRPO algorithm in the DeepSeekMath paper https://arxiv.org/abs/2402.03300.
    """

    def __init__(
            self,
            reward_model: Reward,
            policy_ref: Policy,
            lm_model,  # language model
            tokenizer: GPT2Tokenizer,
            config: dict,
            train: bool = True,
            device: int = 0  # index of current device
    ):
        super().__init__(lm_model, tokenizer, config, train, device)
        self.reward_model = reward_model
        self.policy_ref = policy_ref  # reference policy
        self.truncate_tokens = [self.tokenizer.encode('\n')[0], self.tokenizer.eos_token_id]  # '\n' is 198

    def kl_divergence(self, logp, logp_ref):
        if torch.allclose(logp, logp_ref):
            return torch.zeros_like(logp)
        else:
            kl = torch.exp(logp_ref - logp) - (logp_ref - logp) - 1
            #assert kl >= 0, "kl-divergence is non-negative"
            # some kl are negative though very small, like 0.01.
            return torch.maximum(kl, torch.zeros_like(kl, device=self.device, dtype=torch.bfloat16))

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
        return processed

    def _filter_response(self, responses):
        masks = []
        for t in self.truncate_tokens:
            masks.append(torch.eq(responses, t).int())  # like 0, 0, ..., 1, 0, ...0, 1, 0,..
        mask = torch.maximum(*masks)
        return torch.any(mask, dim=1)

    @torch.no_grad()
    def compute_rewards(self, prompt, responses, n_samples: int):
        # post process responses
        processed = self._post_process(responses)
        # call reward_model to compute rewards
        reward_input = torch.cat((prompt, processed), dim=1)  # prompt left padded, response right padded.
        raw_rewards = self.reward_model.compute_reward(reward_input)  # shape=(n * n_samples,)
        # normalize rewards per group
        rewards = raw_rewards.reshape(-1, n_samples)  # shape=(n, n_samples)
        group_mean, group_std = rewards.mean(dim=1, keepdim=True), rewards.std(dim=1, keepdim=True)
        rewards = ((rewards - group_mean) / group_std).reshape(-1)  # shape=(n * n_samples,)
        # penalize reward
        valid_mask = self._filter_response(responses)  # shape=(n * n_samples,)
        rewards = torch.where(valid_mask, rewards, self.config['penalty_reward_value'])  # shape=(n * n_samples,)
        return rewards, processed

    def rejection_sampling(self, responses, rewards, logps, beta=0.1):
        """
        Algorithm 1 (conduct_rejection_sampling function) from paper https://arxiv.org/abs/2309.06657.

        responses: responses to a prompt
        rewards: rewards of these responses, in the same order as responses
        num_samples: number to select
        beta: beta parameter in KL-constrained reward maximization objective

        return accepted sample responses and their rewards
        """
        candidates = {c: (r, logp) for c, r, logp in zip(torch.unbind(responses), torch.unbind(rewards), torch.unbind(logps))}
        accepted = []
        while len(accepted) < self.config['n_samples_select']:
            max_reward = max(candidates.values(), key=itemgetter(0))[0]
            to_remove = []
            for c, (r, logp) in candidates.items():
                u = torch.rand(1, dtype=torch.bfloat16, device=self.device)
                if u >= torch.exp((r - max_reward) / beta):  # todo: range of right side value?
                    continue
                accepted.append((c, r, logp))
                to_remove.append(c)
                if len(accepted) == self.config['n_samples_select']:
                    break
            for c in to_remove:
                candidates.pop(c)

        return (torch.stack([t[0] for t in accepted]),
                torch.stack([t[1] for t in accepted]),
                torch.stack([t[2] for t in accepted]),
                )

    @torch.no_grad()
    def generate_rollouts(self, prompt):
        """
        Generate a number of responses, rewards, logps for given prompts.

        :param prompt: left padded!  shape=(n, prompt_length)
        """
        if prompt.dim() == 1:
            prompt = prompt.view(1, -1)
        n, prompt_length = prompt.shape

        # first, sampling responses
        # responses.shape = logps.shape =(n * n_samples, response_length)
        n_samples = self.config['n_samples']
        responses, logps = self.generate(prompt, n_samples, return_logp=True)

        # second, compute rewards
        rewards, processed_response = self.compute_rewards(prompt.tile((1, n_samples)).reshape(n * n_samples, -1),
                                                           responses, n_samples)
        logps = logps * (processed_response != self.tokenizer.pad_token_id).int()  # ignore logps on padding token

        # try use post processed response. rather than the generated responses
        # if we use the processed response to compute a reward score, we don't care log prob after the truncate token.
        resps, rs, ls = [], [], []
        for i in range(0, len(responses), n_samples):
            resp, r, l = self.rejection_sampling(processed_response[i:i+n_samples], rewards[i:i+n_samples], logps[i:i+n_samples])
            resps.extend(resp)
            rs.extend(r)
            ls.extend(l)
        processed_response, rewards, logps = torch.stack(resps), torch.stack(rs), torch.stack(ls)

        # third, compute reference policy's log probs.
        n_select = self.config['n_samples_select']
        input_ids = torch.cat((prompt.tile((1, n_select)).reshape(n * n_select, -1), processed_response), dim=1)
        logp_ref, _ = self.policy_ref.log_prob(input_ids, prompt_length, avg_seq=False, scale=True,
                                               top_p=self.config['top_p'])  # shape=(n * n_select, response_length)

        return {
            'responses': processed_response,  # shape=(n * n_select, response_length)
            'rewards': rewards.unsqueeze(1),  # shape=(n * n_select, 1)
            'logp': logps,  # current iteration policy log probs. shape=(n * n_select, response_length)
            'logp_ref': logp_ref  # reference policy log probs. shape=(n * n_select, response_length)
        }

    def compute_loss(self, prompts, responses, rewards, old_logps, logp_ref, prompt_length):
        # with torch.no_grad():
        #     logp_ref, _ = self.policy_ref.log_prob(torch.cat((prompts, responses), dim=1), prompt_length,
        #                                 avg_seq=False, scale=True, top_p=self.config['top_p'])
        logp, entropy = self.log_prob(torch.cat((prompts, responses), dim=1), prompt_length,
                                      avg_seq=False, scale=True, return_entropy=True, top_p=self.config['top_p'])
        ratio = torch.exp(logp - old_logps)
        rewards = rewards.view(-1, 1)
        pg_gain = rewards * ratio
        pg_gain_clipped = rewards * torch.clamp(ratio, 1.0 - self.config['cliprange'], 1.0 + self.config['cliprange'])
        pg_gain_min = torch.minimum(pg_gain, pg_gain_clipped)  # true whether reward is positive or negative
        kl = self.kl_divergence(logp, logp_ref)
        loss = self.config['kl_coef'] * kl - pg_gain_min  # per token loss

        # ignore loss on padding tokens, and no preference for longer responses
        response_mask = (responses != self.tokenizer.pad_token_id).int()
        loss = (loss * response_mask).sum() / response_mask.sum()

        clipped = (pg_gain > pg_gain_clipped).float()
        clipped_frac = (clipped * response_mask).sum() / response_mask.sum()
        return (loss,
                {'grpo/kl_penalty': torch.mean(kl),
                 'grpo/entropy': torch.mean(entropy),
                 'grpo/clip_frac': clipped_frac
                 })

    @torch.no_grad()
    def eval_accuracy(self, input_ids, prompt_length):
        logp_ref, _ = self.policy_ref.log_prob(input_ids, prompt_length)
        logp, _ = self.log_prob(input_ids, prompt_length)

        loss, reward_w, reward_l = self.compute_loss_reward(logp, logp_ref)
        return loss, {
            f"reward/{self.config['model_for']}_eval_chosen": reward_w.mean(),
            f"reward/{self.config['model_for']}_eval_rejected": reward_l.mean(),
            f"reward/{self.config['model_for']}_eval_acc": (reward_w > reward_l).float().mean()
        }


from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

dataset_cache = 'cached_dataset'
Path(dataset_cache).mkdir(parents=True, exist_ok=True)


class TldrDataset:
    def __init__(self, name: str, tokenizer, max_prompt_length: int = 512, max_response_length: int = 64):
        self.train = load_dataset(name, split='train', cache_dir=dataset_cache)
        self.val = load_dataset(name, split='validation', cache_dir=dataset_cache)
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.tensor_dtype = torch.int32

    def __len__(self):
        return len(self.train)

    def len_val(self):
        return len(self.val)

    def shuffle(self):
        self.train.shuffle()

    def extract_prompt(self, prompt: str):
        """Process an original prompt from dataset

        The returned token has 4 parts: subreddit, title, POST: <post truncated at start> TL;DR:
        """
        post_start, suffix_start = prompt.find('POST:'), prompt.rfind('TL;DR:')
        if post_start == -1:
            post_start = 0
        if suffix_start == -1:
            suffix_start = len(prompt)
        prefix = self.tokenizer.encode(prompt[:post_start]) if post_start > 0 else []  # subreddit and title
        post_prompt = self.tokenizer.encode('POST:')
        suffix = self.tokenizer.encode('TL;DR:')
        max_post_length = self.max_prompt_length - len(prefix) - len(post_prompt) - len(suffix)
        # truncate post content from start if too long
        start = post_start + len('POST:') if post_start > 0 else 0
        post = self.tokenizer.encode(prompt[start: suffix_start])[-max_post_length:]  # truncate start
        return prefix + post_prompt + post + suffix

    def pad(self, tensors, padding_values, padding_sides):
        result = []
        for t, pv, ps in zip(tensors, padding_values, padding_sides):
            result.append(pad_sequence(t, batch_first=True, padding_value=pv, padding_side=ps))
        return result


class TldrCompletion(TldrDataset):
    """Dataset https://huggingface.co/datasets/trl-lib/tldr
    For Supervised fine-tuning a base model
    """

    def __init__(self, tokenizer):
        super().__init__('trl-lib/tldr', tokenizer)

    def get_batch(self, idx: torch.Tensor, train: bool = True, with_completion: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        items = self.train[idx] if train else self.val[idx]
        prompts, completions, prompt_masks, completion_masks = [], [], [], []
        for prompt, completion in zip(items['prompt'], items['completion']):
            prompt_token = self.extract_prompt(prompt)
            prompts.append(torch.as_tensor(prompt_token))
            completion_token = self.tokenizer.encode(completion)[:self.max_response_length-1]  # truncate end if too long
            completion_token.append(self.tokenizer.eos_token_id)  # explicitly require an end token
            completions.append(torch.as_tensor(completion_token))
            prompt_masks.append(torch.ones(len(prompt_token)))
            completion_masks.append(torch.ones(len(completion_token)))

        # now padding
        prompts, prompt_masks, completions, completion_masks = self.pad(
            (prompts, prompt_masks, completions, completion_masks),
            (self.tokenizer.eos_token_id, 0, self.tokenizer.eos_token_id, 0),
            ('left', 'left', 'right', 'right')
        )
        if with_completion:
            input_ids = torch.cat((prompts, completions), dim=1)
            mask = torch.cat((prompt_masks, completion_masks), dim=1)
        else:
            input_ids, mask = prompts, prompt_masks
        return input_ids, mask


class TldrPreference(TldrDataset):
    """Dataset https://huggingface.co/datasets/trl-lib/tldr-preference"""

    def __init__(self, tokenizer):
        super().__init__('trl-lib/tldr-preference', tokenizer)

    def get_batch(self, idx: torch.Tensor, train: bool = True):
        items = self.train[idx] if train else self.val[idx]
        chosens, rejects, chosen_masks, reject_masks = [], [], [], []

        for prompt, chosen, rejected in zip(items['prompt'], items['chosen'], items['rejected']):
            prompt_token = self.extract_prompt(prompt)
            chosen_token = self.tokenizer.encode(chosen)[:self.max_response_length-1]  # truncate end if too long
            chosen_token = torch.as_tensor(prompt_token + chosen_token + [self.tokenizer.eos_token_id])
            chosens.append(chosen_token)
            reject_token = self.tokenizer.encode(rejected)[:self.max_response_length-1]  # truncate end if too long
            reject_token = torch.as_tensor(prompt_token + reject_token + [self.tokenizer.eos_token_id])
            rejects.append(reject_token)

            chosen_masks.append(torch.ones(len(chosen_token)))
            reject_masks.append(torch.ones(len(reject_token)))

        # concatenate chosens and rejects. first half is chosen, second half reject
        chosens.extend(rejects)
        chosen_masks.extend(reject_masks)
        ids, masks = self.pad(
            (chosens, chosen_masks),
            (self.tokenizer.eos_token_id, 0),
            ('left', 'left')
        )
        return ids, masks


if __name__ == '__main__':
    pass

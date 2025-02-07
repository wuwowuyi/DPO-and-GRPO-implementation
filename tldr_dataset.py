from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

dataset_cache = 'cached_dataset'
Path(dataset_cache).mkdir(parents=True, exist_ok=True)


class TldrDataset:
    def __init__(self, name: str, tokenizer, max_prompt_length: int = 768, max_response_length: int = 256):
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


class TldrCompletion(TldrDataset):
    """Dataset https://huggingface.co/datasets/trl-lib/tldr
    For Supervised fine-tuning a base model
    """

    def __init__(self, tokenizer):
        super().__init__('trl-lib/tldr', tokenizer)

    def extract_prompt(self, prompt: str):
        """Process an original prompt from dataset
        """
        post_start, suffix_start = prompt.find('POST:'), prompt.rfind('TL;DR:')
        assert post_start > -1 and suffix_start > -1
        prefix = self.tokenizer.encode(prompt[:post_start])  # contains subreddit and title
        post_prompt = self.tokenizer.encode('POST:')
        suffix = self.tokenizer.encode('TL;DR:')
        max_post_length = self.max_prompt_length - len(prefix) - len(post_prompt) - len(suffix)
        post = self.tokenizer.encode(prompt[post_start + len('POST:'): suffix_start])[-max_post_length:]  # post content, truncate start if too long
        return prefix + post_prompt + post + suffix  # subreddit, title, POST: <post> TL;DR:

    def get_batch(self, idx: torch.Tensor, train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
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
        prompts = pad_sequence(prompts, batch_first=True, padding_value=self.tokenizer.eos_token_id, padding_side='left')
        prompt_masks = pad_sequence(prompt_masks, batch_first=True, padding_value=0, padding_side='left')
        completions = pad_sequence(completions, batch_first=True, padding_value=self.tokenizer.eos_token_id, padding_side='right')
        completion_masks = pad_sequence(completion_masks, batch_first=True, padding_value=0, padding_side='right')

        input_ids = torch.cat((prompts, completions), dim=-1)
        mask = torch.cat((prompt_masks, completion_masks), dim=-1)
        return input_ids, mask


class TldrPreference(TldrDataset):
    """Dataset https://huggingface.co/datasets/trl-lib/tldr-preference"""

    def __init__(self, tokenizer):
        super().__init__('trl-lib/tldr-preference', tokenizer)



if __name__ == '__main__':
    pass


import torch
from torch.nn import functional as F
from transformers import GPT2Tokenizer


class Policy:
    def __init__(
            self,
            lm_model,  # language model
            tokenizer: GPT2Tokenizer,
            config: dict,
            train: bool,
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

import torch
from typing import Union, Optional
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Spam_Collator:

    def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len


    def __call__(self, batch):
        texts = [i['message'] for i in batch]
        labels = [i['is_spam'] for i in batch]

        if self.max_seq_len:
            encodings = self.tokenizer(
                texts,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
        else:
            encodings = self.tokenizer(
                texts,
                add_special_tokens=True,
                return_tensors="pt",
            )
        encodings = encodings.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        return encodings, labels

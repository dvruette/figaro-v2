from typing import List, Optional, Literal

import torch
import torch.nn.functional as F
from transformers import BatchEncoding

from figaro.tokenizers import RemiTokenizer
from figaro.preprocessing.remi import RemiObject


def _fill_with_nearest_nonzero_2d(x):
    """
    Function to fill zero values in a tensor with the nearest non-zero value to the left.

    The algorithm is works by constructind flat indices and flat values:
    Indices have the same shape as the input tensor, but flattened.
    Values only contain the non-zero elements, flattened.
    The i-th index is chosen such that the values[index[i] is the nearest non-zero.
    """

    index = (x != 0).cumsum(dim=1)
    index = index + (index.max(1).values.cumsum(0) - index[0].max()).unsqueeze(1)
    index += torch.arange(x.size(0)).unsqueeze(1)
    index = index.flatten().long()

    x_prime = x.clone()
    x_prime[:, 0] -= (x_prime[:, 0] == 0).int()  # handle edge case where first element is zero
    values = x_prime[x_prime != 0]

    return values.gather(0, index).view(x.shape).clamp(0)


def _fill_with_nearest_nonzero_1d(x):
    index = (x != 0).cumsum(0)
    x_prime = x.clone()
    x_prime[0] -= (x_prime[0] == 0).int()  # handle edge case where first element is zero
    values = x_prime[x_prime != 0]
    return values.gather(0, index).clamp(0)


class RemiCollatorForCausalLM:
    """
    Collator for causal language modeling with Remi.

    Args:
        - tokenizer (RemiTokenizer):
            tokenizer to use
        - max_length (Optional[int]):
            maximum length of the sequence. Defaults to 512.
        - pad_to_multiple_of (Optional[int]):
            pad to multiple of this number. Defaults to None.
        - padding_direction (Literal["right", "left"]):
            pad to the right or left. Defaults to "right".
        - drop_empty (bool):
            Whether or not to drop empty sequences (e.g. corrupted files). Defaults to True.
            If True, can lead to varying batch sizes (an error is raised if a batch has no non-empty samples).
            If False, can lead to empty sequences.
    """
    def __init__(
        self,
        tokenizer: RemiTokenizer,
        max_length: Optional[int] = 512,
        pad_to_multiple_of: Optional[int] = None,
        padding_direction: Literal["right", "left"] = "right",
        drop_empty: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding_direction = padding_direction
        self.drop_empty = drop_empty

        # add bar_ids and position_ids for positional embedding
        self._bar_token_ids = torch.tensor(sorted(self.tokenizer.bar_token_ids))
        self._position_token_ids = torch.tensor(sorted(self.tokenizer.position_token_ids))

        # for position algorithm to work, token ids must be consecutive and sorted
        assert self._bar_token_ids[-1] - self._bar_token_ids[0] == len(self._bar_token_ids) - 1
        assert self._position_token_ids[-1] - self._position_token_ids[0] == len(self._position_token_ids) - 1


    def __call__(
        self,
        examples: List[RemiObject],
    ) -> BatchEncoding:
        encoded = []
        for example in examples:
            remi_str = str(example)
            if self.drop_empty and remi_str == "":
                continue

            tokens = self.tokenizer(
                remi_str,
                max_length=None,
                truncation=False,
                return_tensors="pt",
            )

            def _make_position_tensor(x, token_ids):
                x.squeeze_(0)
                token_mask = (x.unsqueeze(-1) == token_ids).any(-1)
                x = (x - token_ids[0]).masked_fill(~token_mask, 0)
                x = _fill_with_nearest_nonzero_1d(x)
                return x.unsqueeze(0)

            bar_ids = _make_position_tensor(tokens["input_ids"].clone(), self._bar_token_ids)
            position_ids = _make_position_tensor(tokens["input_ids"].clone(), self._position_token_ids)

            tokens["bar_ids"] = bar_ids
            tokens["position_ids"] = position_ids

            if self.max_length is not None and self.max_length + 1 < tokens["input_ids"].shape[1]:
                # extract random segment
                start = torch.randint(0, tokens["input_ids"].size(1) - self.max_length - 1, size=(1,)).item()
                end = start + self.max_length
                tokens["labels"] = tokens["input_ids"][:, start + 1:end + 1].clone()
                tokens["input_ids"] = tokens["input_ids"][:, start:end]
                tokens["attention_mask"] = tokens["attention_mask"][:, start:end]
                tokens["bar_ids"] = tokens["bar_ids"][:, start:end]
                tokens["position_ids"] = tokens["position_ids"][:, start:end]
            else:
                tokens["labels"] = tokens["input_ids"][:, 1:].clone()
                tokens["input_ids"] = tokens["input_ids"][:, :-1]
                tokens["attention_mask"] = tokens["attention_mask"][:, :-1]
                tokens["bar_ids"] = tokens["bar_ids"][:, :-1]
                tokens["position_ids"] = tokens["position_ids"][:, :-1]

            encoded.append(tokens)

        if len(encoded) == 0:
            raise ValueError(
                "Encountered batch with only empty/invalid samples. "
                "Consider cleaning your dataset (make sure your MIDI files can be loaded with `pretty_midi`)."
            )

        # pad to sequences to same length
        max_length = max([x["input_ids"].shape[1] for x in encoded])
        if self.pad_to_multiple_of is not None:
            if self.pad_to_multiple_of <= 0:
                raise ValueError("`pad_to_multiple_of` must be a positive integer or None")
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        
        batch = encoded[0]
        for key in batch.keys():
            pad_value = {
                "attention_mask": 0,
                "labels": -100,
            }.get(key, self.tokenizer.pad_token_id)

            tensors = []
            for enc in encoded:
                x = enc[key]
                if self.padding_direction == "right":
                    padding = (0, max_length - x.shape[1])
                else:
                    padding = (max_length - x.shape[1], 0)
                tensors.append(F.pad(x, padding, value=pad_value))
            batch[key] = torch.concatenate(tensors)

        return batch

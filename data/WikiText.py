from datasets import load_dataset
import torch
from typing import Literal
import tiktoken

_wikitext_subsets_tuple = (
    "wikitext-103-raw-v1",
    "wikitext-2-raw-v1",
    "wikitext-103-v1",
    "wikitext-2-v1",
)


class WikiTextDataset:
    def __init__(
        self,
        subset: Literal[
            "wikitext-103-raw-v1",
            "wikitext-2-raw-v1",
            "wikitext-103-v1",
            "wikitext-2-v1",
        ],
        split: Literal["train", "validation", "test"],
        max_length: int,
        stride: int,
        tokenizer: tiktoken.Encoding,
    ) -> None:
        if subset not in _wikitext_subsets_tuple:
            raise ValueError(f"Subset {subset} is not a valid wikitext subset.")
        ds = load_dataset("Salesforce/wikitext", subset, split=split)
        # Filter out empty or whitespace-only texts
        ds = ds.filter(lambda x: bool(x["text"] and x["text"].strip()))
        all_text = "\n".join(ds["text"])
        token_ids = tokenizer.encode(all_text)
        self.total_tokens = len(token_ids)

        self.input_ids = []
        self.target_ids = []
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]

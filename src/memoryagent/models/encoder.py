from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BGEEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        use_query_instruction: bool = True,
        query_instruction: str = "Represent this sentence for searching relevant passages: ",
        max_length: int = 512,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.use_query_instruction = use_query_instruction
        self.query_instruction = query_instruction
        self.max_length = max_length

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    def encode(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        max_length: int | None = None,
    ) -> torch.Tensor:
        """Tokenize → forward → CLS pool → L2-normalize. Returns [N, dim].

        The caller owns the grad context; wrap in torch.no_grad() for inference.
        """
        if is_query and self.use_query_instruction:
            texts = [self.query_instruction + t for t in texts]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length or self.max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return F.normalize(cls, dim=-1)

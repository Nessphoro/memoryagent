from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenGenerator(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        *,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        device_map: str | None = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._needs_manual_final_norm = self._verify_hidden_state_extraction()

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def _find_final_norm(self) -> nn.Module | None:
        # HF convention: model.model.norm; some hybrids may use a different name
        inner = getattr(self.model, "model", self.model)
        for attr in ("norm", "final_layernorm", "final_norm"):
            mod = getattr(inner, attr, None)
            if mod is not None:
                return mod
        return None

    def _verify_hidden_state_extraction(self) -> bool:
        """Returns True iff `hidden_states[-1]` is pre-final-norm and we must
        apply the norm manually to recover the post-norm representation that
        `lm_head` consumes. Uses `lm_head` as ground truth: whichever of
        `lm_head(h_last)` or `lm_head(norm(h_last))` reproduces `out.logits`
        tells us where in the pipeline `hidden_states[-1]` sits.
        """
        final_norm = self._find_final_norm()
        if final_norm is None:
            return False
        device = next(self.model.parameters()).device
        dummy = torch.tensor([[1, 2, 3]], device=device)
        with torch.no_grad():
            out = self.model(dummy, output_hidden_states=True, use_cache=False)
            h_last = out.hidden_states[-1]
            logits_raw = self.model.lm_head(h_last)
            logits_normed = self.model.lm_head(final_norm(h_last))
        atol = 1e-2 if h_last.dtype == torch.bfloat16 else 1e-4
        rtol = 1e-2 if h_last.dtype == torch.bfloat16 else 1e-4
        if torch.allclose(out.logits, logits_raw, atol=atol, rtol=rtol):
            return False
        if torch.allclose(out.logits, logits_normed, atol=atol, rtol=rtol):
            return True
        raise RuntimeError(
            "hidden_states[-1] matches neither raw nor normed input to lm_head; "
            "manual hidden-state extraction needed."
        )

    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Last-token post-final-RMSNorm hidden state. Returns [B, hidden_size]."""
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        h = out.hidden_states[-1]
        if self._needs_manual_final_norm:
            final_norm = self._find_final_norm()
            assert final_norm is not None
            h = final_norm(h)
        last_idx = attention_mask.sum(dim=1).long() - 1
        return h[torch.arange(h.size(0), device=h.device), last_idx]

    def lm_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

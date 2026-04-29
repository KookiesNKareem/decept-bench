"""HF transformers backend mirroring AnthropicBatchClient interface.

Used for models that vLLM 0.19.1 doesn't support (e.g., Gemma 4 with
head_dim=512). Slower than vLLM but always works.

Usage matches decept_api_client.AnthropicBatchClient:
  client = HFBatchClient(model="google/gemma-4-E4B-it", batch_size=8)
  outs = await client.generate([{"messages": [...]}, ...], max_tokens=384)
"""
from __future__ import annotations

import asyncio
import os
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def _resolve_model_class(model_name_or_path: str):
    """Pick the right model class based on architecture."""
    cfg = AutoConfig.from_pretrained(model_name_or_path)
    arches = cfg.architectures or []
    if any("Gemma4" in a for a in arches):
        from transformers import Gemma4ForConditionalGeneration
        return Gemma4ForConditionalGeneration
    return AutoModelForCausalLM


def _resolve_local_path(model_name: str) -> str:
    """If model is in HF cache, return the local snapshot path. Else download
    via snapshot_download and return that path."""
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(model_name)
        return path
    except Exception:
        return model_name


class HFBatchClient:
    def __init__(self, model: str, batch_size: int = 8,
                 attn_impl: str = "eager", torch_dtype=torch.bfloat16,
                 device_map: str = "cuda"):
        print(f"loading HF model {model} (attn={attn_impl})...")
        local_path = _resolve_local_path(model)
        if local_path != model:
            print(f"  using local cache path: {local_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        ModelClass = _resolve_model_class(local_path)
        print(f"  using {ModelClass.__name__}")
        self.model_obj = ModelClass.from_pretrained(
            local_path, torch_dtype=torch_dtype, device_map=device_map,
            attn_implementation=attn_impl,
        )
        self.model_obj.eval()
        self.batch_size = batch_size
        self.total_input_tok = 0
        self.total_output_tok = 0
        self.model_name = model
        print(f"  loaded ({sum(p.numel() for p in self.model_obj.parameters())/1e9:.2f}B params)")

    def _format_prompt(self, p: dict) -> str:
        """Convert {messages, system?, prefill?} dict into a single prompt string."""
        msgs = list(p.get("messages", []))
        sys_p = p.get("system")
        prefill = p.get("prefill")
        if sys_p:
            msgs = [{"role": "system", "content": sys_p}] + msgs
        try:
            prompt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=(prefill is None))
        except Exception:
            # Some models don't have a chat template; concat naively
            parts = []
            for m in msgs:
                parts.append(f"{m['role']}: {m['content']}")
            prompt = "\n\n".join(parts) + "\nassistant:"
        if prefill:
            prompt = prompt + prefill
        return prompt

    @torch.no_grad()
    def _generate_batch(self, prompts: List[str], max_tokens: int,
                          temperature: float) -> List[str]:
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                              truncation=True, max_length=2048).to(self.model_obj.device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-5),
            use_cache=True,
        )
        gen = self.model_obj.generate(**enc, **gen_kwargs)
        new = gen[:, enc["input_ids"].shape[1]:]
        outs = self.tokenizer.batch_decode(new, skip_special_tokens=True)
        self.total_input_tok += int(enc["input_ids"].numel())
        self.total_output_tok += int(new.numel())
        return outs

    async def generate(self, prompts: List[Dict], max_tokens: int = 384,
                         temperature: float = 0.0) -> List[str]:
        """Match AnthropicBatchClient API. Synchronous under the hood, but
        wrapped in asyncio for drop-in compat."""
        formatted = []
        prefill_strs = []
        for p in prompts:
            formatted.append(self._format_prompt(p))
            prefill_strs.append(p.get("prefill"))
        outs = []
        for i in range(0, len(formatted), self.batch_size):
            chunk = formatted[i:i + self.batch_size]
            chunk_outs = self._generate_batch(chunk, max_tokens, temperature)
            outs.extend(chunk_outs)
            await asyncio.sleep(0)  # cooperate with event loop
        # If prefill was requested, prepend it (so callers can extract from full text)
        final = []
        for o, pf in zip(outs, prefill_strs):
            final.append((pf + o) if pf else o)
        return final

    def cost_estimate_usd(self, **kwargs) -> float:
        return 0.0  # local model

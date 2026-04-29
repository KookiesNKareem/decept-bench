"""vLLM batch client mirroring AnthropicBatchClient async interface.

Used for fast local generation with vLLM. Compatible with the run_m* functions
defined in decept_mechanisms.py.

Includes optional output post-processing (e.g., for DSR1's BPE artifacts and
<think> tags).
"""
from __future__ import annotations

import asyncio
import re
import sys
from typing import Callable, Dict, List, Optional


from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def clean_dsr1(text: str) -> str:
    """Strip BPE artifacts and extract post-think content."""
    if not text: return ""
    cleaned = text.replace("Ġ", " ").replace("Ċ", "\n")
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1]
    return cleaned.strip()


class VLLMBatchClient:
    def __init__(self, model: str, max_model_len: int = 4096,
                 gpu_mem: float = 0.85,
                 post_process: Optional[Callable[[str], str]] = None):
        print(f"loading vLLM with {model}...")
        self.llm = LLM(model=model, dtype="bfloat16",
                        gpu_memory_utilization=gpu_mem,
                        max_model_len=max_model_len)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.post_process = post_process or (lambda x: x)
        self.total_input_tok = 0
        self.total_output_tok = 0

    def _format_prompt(self, p: dict) -> str:
        msgs = list(p.get("messages", []))
        sys_p = p.get("system")
        prefill = p.get("prefill")
        if sys_p:
            msgs = [{"role": "system", "content": sys_p}] + msgs
        try:
            prompt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=(prefill is None))
        except Exception:
            parts = []
            for m in msgs:
                parts.append(f"{m['role']}: {m['content']}")
            prompt = "\n\n".join(parts) + "\nassistant:"
        if prefill:
            prompt = prompt + prefill
        return prompt

    async def generate(self, prompts: List[Dict], max_tokens: int = 384,
                         temperature: float = 0.0) -> List[str]:
        formatted = [self._format_prompt(p) for p in prompts]
        prefills = [p.get("prefill") for p in prompts]
        sp = SamplingParams(temperature=temperature, top_p=1.0,
                              max_tokens=max_tokens)
        # vLLM batch generation is sync, but we wrap in asyncio for compat
        loop = asyncio.get_event_loop()

        def _do_gen():
            outs = self.llm.generate(formatted, sp)
            return [o.outputs[0].text for o in outs]

        raw_outs = await loop.run_in_executor(None, _do_gen)
        # Post-process (e.g., DSR1 cleaning) and prepend prefills
        final = []
        for o, pf in zip(raw_outs, prefills):
            cleaned = self.post_process(o)
            if pf:
                cleaned = pf + cleaned
            final.append(cleaned)
        # Tokens: vLLM doesn't expose easily here; estimate
        self.total_output_tok += sum(len(o.split()) * 1.3 for o in raw_outs)
        return final

    def cost_estimate_usd(self, **kwargs) -> float:
        return 0.0

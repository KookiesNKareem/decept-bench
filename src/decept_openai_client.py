"""OpenAI batch client mirroring AnthropicBatchClient interface.

Same generate(prompts, max_tokens, temperature) → list[str] API. Tracks
total_input_tok / total_output_tok / cost_estimate_usd.

Prefill: OpenAI Chat Completions does not support assistant-message prefill
the way Anthropic's API does. When a prompt has a 'prefill' key, this client
returns the __PREFILL_UNSUPPORTED__ sentinel — same one the Anthropic client
uses on Sonnet 4.6 — so run_m5's probe path triggers a couldnt_be_tested
result.
"""
from __future__ import annotations

import asyncio
from typing import List, Tuple

try:
    import openai
except ImportError:
    openai = None


# Rough per-1M-token rates as of early 2026. Override via cost_estimate_usd args.
_DEFAULT_RATES = {
    "gpt-5.5":       (1.25, 10.0),
    "gpt-5.5-mini":  (0.25, 2.0),
    "gpt-5.4":       (1.25, 10.0),
    "gpt-5.4-mini":  (0.25, 2.0),
    "gpt-5":         (1.25, 10.0),
    "gpt-5-mini":    (0.25, 2.0),
    "gpt-5-nano":    (0.05, 0.40),
    "gpt-4o":        (2.50, 10.0),
    "gpt-4o-mini":   (0.15, 0.60),
    "o4-mini":       (1.10, 4.40),
    "o3":            (2.00, 8.00),
    "o3-mini":       (1.10, 4.40),
}


def _rate_for(model: str) -> Tuple[float, float]:
    for k, v in _DEFAULT_RATES.items():
        if model.startswith(k):
            return v
    return (1.0, 5.0)


class OpenAIBatchClient:
    def __init__(self, model: str, max_concurrency: int = 8,
                 max_retries: int = 4, reasoning_effort: str = None):
        self.model = model
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        # None or "minimal"/"low"/"medium"/"high"/"xhigh"
        self.reasoning_effort = reasoning_effort
        self.client = openai.AsyncOpenAI()
        self._sem = asyncio.Semaphore(max_concurrency)
        self.total_input_tok = 0
        self.total_output_tok = 0

    async def _one(self, idx: int, messages: list, system: str = None,
                     max_tokens: int = 384, temperature: float = 0.0,
                     prefill: str = None) -> Tuple[int, str]:
        if prefill:
            # OpenAI doesn't support assistant prefill cleanly. Surface the
            # same sentinel as the Anthropic client so M5's probe detects it.
            return idx, "__PREFILL_UNSUPPORTED__"

        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        # Reasoning models (o-series, gpt-5*) often disallow temperature!=1
        # and use max_completion_tokens. Build kwargs accordingly.
        kwargs = dict(model=self.model, messages=msgs)
        is_reasoning = (self.model.startswith("o") or
                        self.model.startswith("gpt-5"))
        if is_reasoning:
            # Reasoning models burn tokens internally before producing
            # visible output. A small max_tokens (e.g. 200) often produces
            # an empty visible answer because the reasoning trace alone
            # consumed the budget. Floor at 2048 for reasoning models.
            kwargs["max_completion_tokens"] = max(max_tokens, 2048)
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                async with self._sem:
                    resp = await self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                if resp.usage:
                    self.total_input_tok += resp.usage.prompt_tokens
                    self.total_output_tok += resp.usage.completion_tokens
                return idx, text
            except (openai.RateLimitError, openai.APIConnectionError,
                    openai.InternalServerError) as e:
                if attempt == self.max_retries - 1:
                    print(f"  retry exhausted on idx={idx}: {e}")
                    return idx, ""
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
            except openai.BadRequestError as e:
                msg = str(e)
                if "prefill" in msg.lower() or "must alternate" in msg.lower():
                    return idx, "__PREFILL_UNSUPPORTED__"
                print(f"  bad request idx={idx}: {e}")
                return idx, ""
        return idx, ""

    async def generate(self, prompts: List[dict], max_tokens: int = 384,
                         temperature: float = 0.0) -> List[str]:
        tasks = [
            self._one(i, p["messages"], p.get("system"),
                      max_tokens, temperature, p.get("prefill"))
            for i, p in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def cost_estimate_usd(self, in_per_M: float = None,
                           out_per_M: float = None) -> float:
        if in_per_M is None or out_per_M is None:
            in_per_M, out_per_M = _rate_for(self.model)
        return (self.total_input_tok / 1e6 * in_per_M +
                self.total_output_tok / 1e6 * out_per_M)

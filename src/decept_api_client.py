"""Anthropic API client for DECEPT-Bench.

Async-batched generation with concurrency control + retries. Drop-in
replacement for vLLM `llm.generate(prompts, sp)` style usage.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import List, Tuple

# Lazy import: tests + scoring don't need anthropic. The runner does.
try:
    import anthropic
except ImportError:
    anthropic = None  # noqa: F841 — set None so module imports cleanly


class AnthropicBatchClient:
    def __init__(self, model: str = "claude-haiku-4-5",
                 max_concurrency: int = 8, max_retries: int = 4):
        if anthropic is None:
            raise ImportError(
                "anthropic SDK is required to run the Anthropic backend. "
                "Install with: pip install anthropic"
            )
        self.model = model
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.client = anthropic.AsyncAnthropic()
        self._sem = asyncio.Semaphore(max_concurrency)
        self.total_input_tok = 0
        self.total_output_tok = 0

    async def _one(self, idx: int, messages: list, system: str = None,
                     max_tokens: int = 384, temperature: float = 0.0,
                     prefill: str = None) -> Tuple[int, str]:
        # If prefill is given, append a partial assistant message
        msgs = list(messages)
        if prefill:
            msgs = msgs + [{"role": "assistant", "content": prefill}]
        kwargs = dict(model=self.model, messages=msgs,
                      max_tokens=max_tokens, temperature=temperature)
        if system:
            kwargs["system"] = system
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                async with self._sem:
                    resp = await self.client.messages.create(**kwargs)
                # Extract text
                text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                if prefill:
                    text = prefill + text
                self.total_input_tok += resp.usage.input_tokens
                self.total_output_tok += resp.usage.output_tokens
                return idx, text
            except (anthropic.RateLimitError,
                      anthropic.APIConnectionError,
                      anthropic.InternalServerError) as e:
                if attempt == self.max_retries - 1:
                    print(f"  retry exhausted on idx={idx}: {e}")
                    return idx, ""
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
            except anthropic.BadRequestError as e:
                msg = str(e)
                if "does not support assistant message prefill" in msg:
                    return idx, "__PREFILL_UNSUPPORTED__"
                print(f"  bad request idx={idx}: {e}")
                return idx, ""
        return idx, ""

    async def generate(self, prompts: List[dict], max_tokens: int = 384,
                         temperature: float = 0.0) -> List[str]:
        """prompts: list of dicts with keys: messages, optional system, optional prefill"""
        tasks = [
            self._one(i, p["messages"],
                      p.get("system"),
                      max_tokens, temperature,
                      p.get("prefill"))
            for i, p in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def cost_estimate_usd(self, in_per_M: float = 1.0,
                           out_per_M: float = 5.0) -> float:
        return (self.total_input_tok / 1e6 * in_per_M +
                self.total_output_tok / 1e6 * out_per_M)


def gen_sync(client: AnthropicBatchClient, prompts: List[dict],
              max_tokens: int = 384, temperature: float = 0.0) -> List[str]:
    """Sync wrapper for callers that don't use asyncio."""
    return asyncio.run(client.generate(prompts, max_tokens, temperature))


if __name__ == "__main__":
    # Quick connectivity test
    c = AnthropicBatchClient(max_concurrency=2)
    out = gen_sync(c, [
        {"messages": [{"role": "user", "content": "Say only the word PONG."}]}
    ], max_tokens=8)
    print("ping:", out)
    print(f"cost: ${c.cost_estimate_usd():.6f}")

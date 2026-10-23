"""Token usage from the response objects of the Gemini, OpenAI-compatible and Anthropic SDKs.

The four numbers that decide the bill are normalised the same way for every provider:

- ``input_tokens``: the whole prompt, cached part included;
- ``cached_input_tokens``: the part of the prompt served from cache;
- ``output_tokens``: visible output, WITHOUT thinking;
- ``thinking_tokens``: reasoning tokens, billed as output.

Each SDK reports them differently, and the differences are where bills go wrong:

- Gemini reports thinking (``thoughts_token_count``) OUTSIDE ``candidates_token_count``.
  Leaving it out under-bills: in the author's crawler, by 74% on average and 11x in the worst
  call.
- OpenAI reports reasoning INSIDE ``completion_tokens``. Adding it again over-bills.
- DeepSeek speaks the OpenAI dialect but reports cache hits as ``prompt_cache_hit_tokens``.
- Anthropic's ``input_tokens`` EXCLUDES cache reads and cache writes, which come in their own
  fields.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    thinking_tokens: int = 0
    provider: str = ""

    @property
    def billed_output(self) -> int:
        return self.output_tokens + self.thinking_tokens

    def __bool__(self) -> bool:
        return any((self.input_tokens, self.output_tokens, self.cached_input_tokens, self.thinking_tokens))


def _n(obj, name) -> int:
    v = getattr(obj, name, None) if not isinstance(obj, dict) else obj.get(name)
    return int(v or 0)


def _get(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def gemini(resp) -> Usage | None:
    u = _get(resp, "usage_metadata")
    if u is None or _get(u, "prompt_token_count") is None:
        return None
    return Usage(_n(u, "prompt_token_count"), _n(u, "candidates_token_count"),
                 _n(u, "cached_content_token_count"), _n(u, "thoughts_token_count"), "google")


def openai(resp) -> Usage | None:
    u = _get(resp, "usage")
    if u is None or _get(u, "prompt_tokens") is None:
        return None
    det_in, det_out = _get(u, "prompt_tokens_details"), _get(u, "completion_tokens_details")
    cached = _n(det_in, "cached_tokens") if det_in else 0
    if not cached:
        cached = _n(u, "prompt_cache_hit_tokens")  # DeepSeek
    reasoning = _n(det_out, "reasoning_tokens") if det_out else 0
    completion = _n(u, "completion_tokens")
    return Usage(_n(u, "prompt_tokens"), max(0, completion - reasoning), cached, reasoning, "openai")


def openai_responses(resp) -> Usage | None:
    """OpenAI Responses API (``object == "response"``, ``input_tokens`` / ``output_tokens``)."""
    u = _get(resp, "usage")
    if u is None or _get(resp, "object") != "response":
        return None
    det_in, det_out = _get(u, "input_tokens_details"), _get(u, "output_tokens_details")
    reasoning = _n(det_out, "reasoning_tokens") if det_out else 0
    return Usage(_n(u, "input_tokens"), max(0, _n(u, "output_tokens") - reasoning),
                 _n(det_in, "cached_tokens") if det_in else 0, reasoning, "openai")


def anthropic(resp) -> Usage | None:
    """Anthropic Messages (``type == "message"``): input_tokens excludes cache reads and writes."""
    u = _get(resp, "usage")
    if u is None or _get(resp, "type") != "message":
        return None
    read, write = _n(u, "cache_read_input_tokens"), _n(u, "cache_creation_input_tokens")
    return Usage(_n(u, "input_tokens") + read + write, _n(u, "output_tokens"), read, 0, "anthropic")


def extract(resp) -> Usage | None:
    """Usage of any supported response, or ``None`` when the object is not recognised."""
    for fn in (gemini, openai, openai_responses, anthropic):
        try:
            u = fn(resp)
        except Exception:
            u = None
        if u is not None:
            return u
    return None

"""One live request per provider.py endpoint. Spends a few cents.

Runs provider.py itself, stdlib only, so it can run on the prod host
without importing the plugin:

    scp plugins/llm/src/llm/{provider,apikeys}.py host:/tmp/smoke/llm/
    touch /tmp/smoke/llm/__init__.py
    python3 provider_smoke.py /tmp/smoke /home/vibebot/.config/vibebot/env [EDIT_IMAGE_URL]

Prints status, usage and cost per call, never a key. Checks what the unit
tests cannot: the providers accept the requests (User-Agent included),
whether xAI chat and responses report cost_in_usd_ticks, whether xAI's
completion_tokens already include reasoning_tokens, and whether a repeated
Haiku call reads the prompt cache.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

sys.path.insert(0, sys.argv[1])
from llm import provider  # noqa: E402

for line in open(sys.argv[2], encoding="utf-8"):  # noqa: PTH123, SIM115
    name, sep, value = line.strip().partition("=")
    if sep and name in ("ANTHROPIC_API_KEY", "XAI_API_KEY"):
        os.environ[name] = value.strip().strip("'\"")
EDIT_URL = sys.argv[3] if len(sys.argv) > 3 else None

raw: list[dict[str, Any]] = []
_real_post = provider._post


def _recording_post(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = _real_post(*args, **kwargs)
    raw.append(data)
    return data


provider._post = _recording_post

HAIKU = "anthropic/claude-haiku-5-5"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
# Well past the minimum cacheable prefix.
SYSTEM = "You are a terse IRC bot. " + " ".join(
    f"Rule {i}: answer in one short line and never use markdown." for i in range(600)
)


def run(label: str, call: Any) -> Any:
    raw.clear()
    t0 = time.monotonic()
    try:
        result = call()
    except provider.ProviderError as exc:
        print(f"FAIL {label}: {type(exc).__name__}: {str(exc)[:300]}")
        return None
    ms = (time.monotonic() - t0) * 1000
    print(f"ok   {label} {ms:.0f} ms")
    usage = raw[-1].get("usage") if raw else None
    print(f"     raw usage: {usage}")
    return result


def show(reply: provider.Reply | None) -> None:
    if reply is None:
        return
    print(
        f"     text={reply.text[:80]!r} tool_calls={[c.name for c in reply.tool_calls]} "
        f"finish={reply.finish_reason} usage={reply.usage} cost=${reply.cost:.6f}"
    )


def xai_reasoning_check(model: str) -> None:
    """Does completion_tokens already count reasoning? Compare the ticks
    with both readings of the price table."""
    usage = (raw[-1].get("usage") or {}) if raw else {}
    ticks = usage.get("cost_in_usd_ticks")
    print(f"     cost_in_usd_ticks present: {ticks is not None}")
    if not ticks:
        return
    price = provider.PRICES[model]
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    # Chat reports it under completion_tokens_details, Responses under
    # output_tokens_details.
    out_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    reasoning = out_details.get("reasoning_tokens") or 0
    base = ((prompt - cached) * price.input + cached * price.cache_read) / 1e6
    print(f"     reasoning_tokens={reasoning} (too few and the two readings agree)")
    print(
        f"     ticks=${ticks * provider.XAI_TICK_USD:.6f}  "
        f"table(completion+reasoning)=${base + (completion + reasoning) * price.output / 1e6:.6f}  "
        f"table(completion only)=${base + completion * price.output / 1e6:.6f}"
    )


print("== Anthropic messages: tool + cached system, twice")
for attempt in (1, 2):
    show(
        run(
            f"haiku #{attempt}",
            lambda: provider.chat(
                HAIKU,
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "What's the weather in Toronto?"},
                ],
                timeout=60,
                tools=TOOLS,
            ),
        )
    )

print("== xAI chat completions")
for model in ("xai/grok-4.3", "xai/grok-4-1-fast-reasoning"):
    show(
        run(
            model,
            lambda model=model: provider.chat(
                model,
                [
                    {
                        "role": "user",
                        "content": "A train leaves at 09:47 and averages 83 km/h over 417 km, "
                        "stopping 3 times for 11 minutes each. At what time does it arrive? "
                        "Answer with just the time.",
                    }
                ],
                timeout=90,
                cache_key="chan:#smoke:main",
            ),
        )
    )
    xai_reasoning_check(model)

show(
    run(
        "grok-4-1-fast-non-reasoning json",
        lambda: provider.chat(
            "xai/grok-4-1-fast-non-reasoning",
            [{"role": "user", "content": 'Reply with the JSON object {"ok": true}.'}],
            timeout=60,
            json_reply=True,
        ),
    )
)

print("== xAI responses + web_search")
reply = run(
    "search",
    lambda: provider.search(
        "xai/grok-4.3", "What is today's weather in Toronto, Ontario?", timeout=120
    ),
)
show(reply)
if reply is not None:
    print(f"     grounded={reply.grounded}")
    xai_reasoning_check("xai/grok-4.3")

print("== xAI images")
picture = run(
    "generations",
    lambda: provider.draw("xai/grok-imagine-image", "a red cube on a white table", timeout=120),
)
if picture is not None:
    print(f"     bytes={len(picture.data or b'')} cost=${picture.cost:.4f}")
if EDIT_URL:
    picture = run(
        "edits",
        lambda: provider.draw(
            "xai/grok-imagine-image", "make it blue", timeout=120, edit_url=EDIT_URL
        ),
    )
    if picture is not None:
        print(f"     bytes={len(picture.data or b'')} cost=${picture.cost:.4f}")
else:
    print("skip edits (no EDIT_IMAGE_URL given)")

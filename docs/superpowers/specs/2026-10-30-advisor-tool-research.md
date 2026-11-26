# Advisor tool: letting Haiku escalate to Sonnet or Opus

Status: implemented for chat turns (`LLM.advisorModel`, default Opus 5.5), skipping the measurement probe. Fixes 1 to 5 below are in `provider.py` and `service.py`; the step timeout for a turn that may consult is 120 seconds.

Sources: the live [advisor tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool), read 2026-10-30, and `plugins/llm/src/llm/provider.py` / `service.py` at `04d3f886`.

## What the advisor tool is

A server-side tool (beta, header `advisor-tool-2026-03-01`) that pairs the request's model (the *executor*) with a stronger *advisor* model:

```json
{"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-5-5",
 "max_uses": 1, "max_tokens": 1024}
```

The executor decides when to call it, like any tool. When it does, Anthropic runs a separate inference on the advisor model inside the same `/v1/messages` request. The advisor sees the executor's whole transcript: system prompt, tool definitions, prior turns, tool results, and the text written so far. It has no tools of its own. Its advice comes back as an `advisor_tool_result` block and the executor carries on writing. No extra round trip on our side, apart from a `pause_turn` resume described later.

That is the "Haiku decides whether it needs help" behaviour, built in. The tool includes its own description that nudges the executor to call it near the start of complex tasks and when stuck.

### Pairing with `claude-haiku-5-5`

Both of the models in question are valid advisors for a Haiku 5.5 executor:

| Advisor | Rate (in/out per MTok) | Advice payload |
|---|---|---|
| `claude-sonnet-5-5` | $2 / $10 | encrypted (`advisor_redacted_result`) |
| `claude-opus-5-5` | $4 / $20 | encrypted |
| `claude-opus-4-8` | $5 / $25 | plaintext (`advisor_result`) |

Encrypted advice can't be read, only replayed: the server decrypts it into the executor's prompt on the next step. For an evaluation phase where we want to log what the advisor said, `claude-opus-4-8` is a valid Haiku 5.5 advisor and returns plaintext.

### Knobs

- `max_uses`: per-request cap on advisor calls. Past it, the result is `max_uses_exceeded` and the executor continues without advice.
- `max_tokens`: caps advisor output (thinking plus text), minimum 1024. The request's own `max_tokens` does **not** bound the advisor.
- `caching`: caches the advisor's prompt across calls. Breaks even at about three calls per conversation, so leave it off here.
- No conversation-level cap. Count calls ourselves and drop the tool from `tools` when over budget. The live docs say the `advisor_tool_result` blocks can stay in history when the tool is removed. (The bundled `claude-api` skill notes say this returns a 400; the live docs contradict it, and they win.)
- Shortening advice: a line addressed to the advisor in the user message works best, for example `(Advisor: please keep your guidance under 80 words.)`.

### Cost per call

Advisor tokens are billed at the advisor's rates. The docs put a typical call at 400 to 700 text tokens, 1,400 to 1,800 including thinking. With a 5K-token transcript as input:

| Advisor | Approximate cost per call |
|---|---|
| Sonnet 5.5 | 5K × $2 + 1.6K × $10 ≈ **$0.026** |
| Opus 5.5 | 5K × $4 + 1.6K × $20 ≈ **$0.052** |

A cached Haiku 5.5 step on the same transcript costs well under a tenth of a cent. One advisor call is roughly 50 to 100 Haiku turns. The docs' own framing: "expect higher cost than Haiku alone, but lower than switching the executor to a larger model." That holds only if Haiku calls it rarely.

## Fit for vibebot

The docs call the advisor "a weaker fit for single-turn Q&A (nothing to plan)". Most `@ask` traffic and channel chat is exactly that. They also note Haiku applies the default guidance conservatively on research and lookup work, which here is a feature: it should rarely call it for chat.

Where it plausibly pays off:

- Multi-step tool chains in the `chat` profile (the planner loop: search, then reminder or scheduled task creation, then a reply).
- Scheduled LLM tasks, which run unattended and can't be corrected by a follow-up message.
- Hard one-off questions where Haiku currently answers confidently and wrong. Unknown how often Haiku would recognise these.

Probably not worth it on `verse` (prose quality, not planning), memory extraction (never shown), or `draw`.

Suggested shape: an optional per-channel `advisorModel` registry value (empty means off), offered only on the `chat` profile, with `max_uses: 1` and `max_tokens: 1024`.

## What the current code would get wrong

In order of how badly each would bite.

1. **Advice silently dropped.** `_anthropic_reply` keeps only `text`, `tool_use`, and `thinking`/`redacted_thinking` blocks. `server_tool_use` and `advisor_tool_result` would be discarded, and `Reply.turn()` rebuilds the assistant turn without them. On a tool-loop step after the advisor ran, the executor would lose the advice. The `thinking_blocks` passthrough is the right pattern; it needs generalising to "opaque Anthropic blocks replayed verbatim, in order". `_xai_request` already strips `thinking_blocks`, so the same treatment keeps provider switching safe.
2. **`pause_turn` delivered as a reply.** `_ANTHROPIC_FINISH` has no `pause_turn` entry, so it maps to `stop`. A response that ends while an advisor call is pending has a `server_tool_use` and no result. Today that would deliver empty or half-written text to the channel. The loop must append the turn unchanged and resend with the same tool and header.
3. **Timeout.** `LLM.timeout` defaults to 30 seconds and the transport is non-streaming. The advisor runs synchronously inside the request; an Opus call producing about 1.6K tokens with thinking, plus the executor's own work, may not fit. Measure with one real call before giving any route the tool. Streaming wouldn't help: the advisor sub-inference doesn't stream either. The levers are the timeout and the tool's `max_tokens`.
4. **Advisor spend books as $0.** Cost is priced from the top-level `usage`, which counts executor tokens only. Advisor tokens are reported in `usage.iterations[]` entries with `type: "advisor_message"` and a `model` field. Pricing needs to walk `iterations` and price those entries against `PRICES["anthropic/" + model]`.
5. **Plumbing.**
   - No way to send an `anthropic-beta` header; `chat()` sets only `x-api-key` and `anthropic-version`.
   - `_anthropic_tool` converts OpenAI function tools only; the advisor needs a raw Anthropic tool definition passed through.
   - Config validation should reject invalid executor/advisor pairs. The API returns a 400 naming the pair, but catching it at `config` time is friendlier.
   - Rate limits: an advisor 429 comes back as `too_many_requests` inside the result and the request still succeeds. Worth a log line, not an error.

Cross-turn history is not a problem: `context.py` stores reply text, so advisor blocks never reach later user turns.

## Alternative: our own `escalate` tool

A client tool that, when Haiku calls it, has the bot send the transcript to Sonnet or Opus itself and return the answer as a `tool_result`.

- For: plaintext advice we can log, works with an xAI executor too, no beta dependency, full control over the transcript sent.
- Against: an extra round trip and a second request's latency, and we write the transcript-forwarding and prompting ourselves. The advisor tool's tuned built-in description and its server-side advisor system prompt are lost.

The advisor tool is less code given fixes 1 to 5, so prefer it unless xAI channels need escalation too.

## Suggested next step

One throwaway script, outside the plugin, sending a real `#channel`-sized transcript to `claude-haiku-5-5` with an Opus 5.5 advisor (and again with Opus 4.8 for readable advice). Record wall time, whether Haiku calls the advisor on a few typical and a few hard prompts, and `usage.iterations`. That settles the timeout and call-rate questions before touching `provider.py`.

# LiteLLM Replacement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Revised during implementation (2026-10-21):** no SDKs. `llm/provider.py` speaks to Anthropic and xAI with stdlib `urllib`, the house idiom (`statuspage`, the old `_xai_image`). Its interface is three calls (`chat`, `search`, `draw`), two result types (`Reply`, `Picture`) and one error family. Tasks 3 and 4 below describe the SDK version and are kept only as the record of the plan. The self-hosted image box is unreachable, so the image fallback and its seven config keys were removed instead of ported; Gemini paths, `frequency_penalty`, `drop_params` and the server-header probe were deleted too.

**Goal:** Remove `litellm` from the dependency tree. Anthropic and xAI are called through one small module that we own.

**Architecture:** A new `llm/providers.py` exposes the four calls `service.py` makes today: `completion`, `responses`, `image_generation` and `completion_cost`. It also exposes the exception classes the code catches. Each call returns OpenAI-shaped objects, which is the shape the code already reads. Behind it there are three backends: `anthropic` (Messages API), xAI (`openai` SDK, `base_url=https://api.x.ai/v1`), and the self-hosted box (`openai` SDK, configured `api_base`). Cost comes from our own price table.

**Tech Stack:** Python 3.14, stdlib `urllib` + `json`, pytest. Removing litellm drops 45 packages from the lock (`openai`, `pydantic`, `httpx`, `tokenizers`, `huggingface-hub`, `aiohttp`, `boto3`, …).

## Why now, and exposure

LiteLLM has been compromised on PyPI twice this year:

- **March:** 1.82.7 and 1.82.8.
- **2026-10-21:** 1.107.0. It was uploaded at 11:39 UTC and later quarantined. Its wheels drop `cook.pth`, `cook.py` and `sitecustomize.py` into site-packages, so the payload runs at interpreter start without `import litellm`. The Docker `latest` tag carried it too.

We were exposed to neither. Checked 2026-10-21:

| Place | litellm | Evidence |
|---|---|---|
| prod container (image built 2026-10-19) | 1.103.0 | `docker exec` listing of site-packages, no `cook*`/`sitecustomize` |
| this Mac (`.venv`, uv cache) | 1.103.0 | `find ~` turned up no 1.107 dist-info and no `cook.pth` (a Limnoria venv holds 1.79.1) |
| `uv.lock` history | 1.81.9 → 1.82.6 → 1.83.14 → … → 1.103.0 | `git show <rev>:uv.lock`; no revision ever held 1.82.7, 1.82.8 or 1.107.0 |
| Dependabot PR #102 (CI ran on a GitHub runner) | 1.104.0 | branch lock; last push 2026-10-09 |
| Docker build | lock only | `uv sync --frozen` / `--locked`, hash-pinned |

The two paths that could have pulled a fresh compromised release were dependabot and a manual `uv lock --upgrade`. Both are now closed by a 7-day cooldown (`[tool.uv] exclude-newer = "7 days"` and dependabot `cooldown.default-days: 7`). Litellm is also in dependabot's ignore list until it is removed. A cooldown would have skipped both incidents.

**Do not rotate keys on account of this incident.** Rotation is warranted only if a host turns up with 1.107.0 installed.

## Why not another aggregator

Prod `bot.conf` names six model strings. They reach exactly three backends:

- `anthropic/claude-haiku-5-5` (assistant, memory, meme, compaction)
- `xai/grok-4.3`, `xai/grok-4-1-fast-{reasoning,non-reasoning}` (search, code, refusal fallback), and `xai/grok-imagine-image` (draw, meme edit)
- `openai//work/models/MiniMax-H3/FL2VA` at the vLLM box (image fallback)

xAI and vLLM speak the OpenAI wire format, so one SDK covers both. Only Anthropic needs translation. An aggregator would sit in the place litellm sits now: another third-party package between us and two vendor SDKs, with its own release cadence to trust.

- **any-llm (Mozilla.ai):** wraps the official SDKs, so we would install those anyway, plus its layer on top.
- **Gateways (Portkey, Bifrost, OpenRouter, Cloudflare):** add either a service to run or a third party that holds our keys.

Neither earns its place for two providers.

The code also already distrusts what litellm adds:

- Its price table missed our image models for four months.
- `-latest` aliases drifted.
- `drop_params` silently discards arguments.
- `_xai_image` already bypasses it for meme edits.

## Global Constraints

- No backwards compatibility (one deployment). Gemini/Vertex branches are deleted, not ported. They are dead in prod: no gemini model in the prod conf since the 2026-10-07 Haiku cutover. The casualty is the local dev `bot.conf` (`askModel`, `drawModel`: gemini; `codeModel: openai/gpt-5.2`), which needs editing to prod-shaped values.
- SDK clients use `max_retries=0`. The pending-tasks backoff ladder and the Timeout→stash contract assume nothing retries underneath them. The timeout is set per client; `litellm.request_timeout = 120` (a Gemini bug workaround) goes away.
- Unknown provider prefix = config validation error. Accepted prefixes are `anthropic/`, `xai/` and `openai/` (the last only with an `api_base`).
- A model missing from the price table logs WARNING once and books $0. This is the same failure mode as today, but loud.
- Before every commit: `make check` green. Commit on `main`; **do not push mid-plan** (push = prod deploy). Push once, after Task 6's smoke run.

## Review Focus

1. **Anthropic parameter hygiene.** litellm's `drop_params=True` hid several mismatches. Confirm each one against a real request in Task 0, before porting:
   - Haiku 5.5 rejects non-default sampling values, but verse sends `temperature` and `frequency_penalty` (service.py:6408-6426, :3759-3769).
   - `reasoning_effort` has no meaning on Anthropic and was a placebo on grok (status_announce :5250).
   - Anthropic requires `max_tokens`, and six call sites never pass one: run_completion, ask_helper, prompt_rewrite, extract_memories, cleanup_memories, and compaction.
2. **`finish_reason` mapping.** `end_turn`→`stop`, `tool_use`→`tool_calls`, `max_tokens`→`length`, `refusal`→a refusal the existing `_is_provider_refusal` path handles. The `!= "stop"` guard at :5280 depends on this.
3. **Thinking blocks round-trip unchanged** (:6826). The adapter must hand back the SDK's blocks verbatim, signature included. It must not rebuild them.
4. **Tool translation.** OpenAI `tools[].function.parameters` maps to `input_schema`. `role=tool` messages become `tool_result` blocks in a single user turn. Assistant `tool_calls` become `tool_use` blocks. `tool_choice` `"required"` maps to `{"type":"any"}`, and the forced-function form maps to `{"type":"tool","name":…}`. Haiku 5.5 still accepts forced choice; only Opus 5.5, Sonnet 5.5 and Fable 5.1 reject it, so note it in the adapter.
5. **Cache markers.** `_with_system_cache_marker` and the top-level `cache_control` move into the adapter as native `system=[{…,"cache_control"}]`. Verify `usage.cache_read_input_tokens` > 0 on the second smoke request.
6. **Structured output.** `response_format` json_schema (extract_memories :8921) maps to Anthropic `output_config={"format": …}`. `json_object` (cleanup_memories, meme_pick) has no Anthropic equivalent, so keep the prompt instruction and parse.

---

### Task 0: Record what litellm actually sends

- [ ] On this Mac, with `LITELLM_LOG=DEBUG`, run one request of each kind against the real APIs and save the request bodies litellm emits to `tmp/litellm-wire/`. The kinds are: assistant step with tools on Haiku, verse turn, memory extract, xAI search, and xAI draw. Cost is about $0.01; get rdrake's OK per the bulk-runs rule anyway.
- [ ] Settle Review Focus 1 from those bodies. They are the fixtures the adapter tests assert against.

### Task 1: The seam (no behaviour change)

**Files:** create `plugins/llm/src/llm/providers.py`; modify `service.py`, `verse/compaction.py`, `tests/conftest.py`.

- [ ] `providers.py` starts as a pass-through to litellm. It exports `completion`, `responses`, `image_generation`, `completion_cost`, and `Timeout`, `RateLimitError`, `AuthenticationError`, `ContentPolicyViolationError`, `BadRequestError`, `NotFoundError`, `APIError`. Keep litellm's constructor shape (`message=, model=, llm_provider=`), since tests build over 40 of these.
- [ ] In `service.py`, change `import litellm` to `from llm import providers as litellm`. All 322 tests that patch `llm.service.litellm.*` keep resolving with zero edits. `make check` green.
- [ ] `compaction.LiteLLMVerseClient` calls `providers.completion`. Rename it to `ProviderVerseClient`.

### Task 2: Own the types and the price table

- [ ] Add the dataclasses `Completion`, `Choice`, `Message`, `ToolCall` and `Usage`, shaped like the fields audited in use: `choices[0].message.content`, `.tool_calls[].id/.function.name/.arguments`, `.thinking_blocks`, `finish_reason`, `usage.prompt_tokens/completion_tokens/cache_read_input_tokens/prompt_tokens_details.cached_tokens`, and `id`. Point conftest's `make_completion_response` at them.
- [ ] `PRICES: dict[str, ModelPrice]` covers the five prod chat models and folds in `IMAGE_COST_PER_IMAGE`. `completion_cost(response, model)` replaces the three call sites (:3229, :4967, compaction:298). The compaction $1 clamp goes, since it existed only because litellm guessed. Delete `register_model` in config.py.
- [ ] `apikeys.provider_of` becomes a prefix split over `PROVIDER_ENV_VARS`. config.py's model validator uses that same prefix plus a `PRICES` membership warning; the difflib suggestions are drawn from `PRICES`. Drop the litellm logger names from the redaction filter and add `anthropic`, `openai` and `httpx`.

### Task 3: OpenAI-shaped backend (xAI + vLLM)

- [ ] `completion` for `xai/…` uses `openai.OpenAI(base_url="https://api.x.ai/v1", api_key=…, max_retries=0, timeout=…)`, one client per key. Pass `extra_headers` / `extra_body` through as-is; that is where `x-grok-conv-id` and `prompt_cache_key` already live.
- [ ] `responses` for `xai/…` calls `client.responses.create(model, input, tools=[{"type":"web_search"}], …)`. The output walk at :4893-4945 stays as is.
- [ ] Draws for `xai/…` route through the existing `_xai_image` (generations endpoint). Draws to an `api_base` (vLLM) use `client.images.generate(..., extra_body={"num_inference_steps"})`. After this task, `image_generation` no longer exists.
- [ ] Map the SDK's exceptions to ours. `APITimeoutError`→`Timeout`. A `BadRequestError` whose body carries xAI moderation text maps to `ContentPolicyViolationError`, using the same keywords `_is_content_safety_error` uses today.

### Task 4: Anthropic backend

- [ ] Add the `anthropic` dependency. `completion` for `anthropic/…` calls `client.messages.create`, translating per Review Focus 2-6. `max_tokens` defaults to 16000 when the caller passes none. Strip whatever Task 0 showed litellm dropping.
- [ ] Map exceptions. `anthropic.APITimeoutError`→`Timeout`. `stop_reason == "refusal"` returns normally with `finish_reason="refusal"`, so `_serve_refusal_fallback` still fires.
- [ ] Adapter tests use the Task 0 bodies. They cover a tool round-trip with thinking blocks, parallel `tool_result`s in one user turn, the system cache marker, and json_schema → `output_config`.

### Task 5: Delete litellm

- [ ] Remove the Gemini/Vertex code: `_GEMINI_SAFETY_SETTINGS`, the grounding tools in `_get_provider_kwargs`/`_resolve_grounding_kwargs`, the `vertex_ai_grounding_metadata` / `_hidden_params` / `model_extra` grounding probes, and `_completion_with_tool_fallback`'s litellm-specific "invalid" retry. Re-check whether the retry is still needed without Gemini.
- [ ] Also remove: `drop_params`, `num_retries`, `metadata`, `reasoning_effort`, and `litellm.request_timeout`.
- [ ] In `tracing.extract_server_headers`, drop the `_response_headers` probe. Get headers from the SDK's `.with_raw_response` only if the DEBUG line is worth keeping; otherwise delete it.
- [ ] Rename the import back: `providers` everywhere. Rewrite the test patch strings in one mechanical pass (`llm.service.litellm.` → `llm.service.providers.`).
- [ ] Remove `litellm` from `plugins/llm/pyproject.toml`, the `pyproject.toml` warning filter, the dependabot ignore entry, and the docs/README "LiteLLM-powered" strings. Run `uv lock` and check that `litellm` and its transitive tree (`tokenizers`, `huggingface-hub`, `tiktoken`, `fastuuid`, …) are gone.

### Task 6: Smoke, then ship

- [ ] Run a local bot against the real APIs: chat with a tool call, verse, memory extract, search, draw, and the vLLM fallback draw. Compare cost lines with `completion_timing` logs from prod for the same kinds of request.
- [ ] Push. The auto-deploy restarts the bot. Watch `completion_timing` and `refusal_fallback_served` for an hour.
- [ ] Prod changes to list in the commit: none in `bot.conf` (all six prod model strings keep their prefixes). The env file is unchanged (`ANTHROPIC_API_KEY`, `XAI_API_KEY`, `ANIMATE_API_KEY`).

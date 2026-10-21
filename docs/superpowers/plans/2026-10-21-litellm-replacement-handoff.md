# LiteLLM replacement: handoff (2026-10-21)

Pick up from here in a fresh session. The original plan is `2026-10-21-litellm-replacement.md`; its Tasks 3–4 (SDKs) are superseded by what is below.

## Where things stand

| Commit | State | What |
|---|---|---|
| `d51cf26` | **pushed, live** | 7-day dependency cooldown (`[tool.uv] exclude-newer = "7 days"`, dependabot `cooldown.default-days: 7`), plan doc. PR #102 (litellm 1.104.0) closed. |
| `033de70` | **local only, NOT pushed** | litellm replaced by `plugins/llm/src/llm/provider.py`. `make check` green: 4243 tests, 93.84% coverage. |

**Do not push `033de70` as is.** A push to `main` auto-deploys to prod within 15 minutes. Do the fixes and the smoke test below first. The new transport has never made a live request: every test mocks `provider._post`, and the wire fixtures are bodies litellm *sent*, not provider replies.

## The incident (closed)

- LiteLLM 1.107.0 (PyPI, 2026-10-21 11:39 UTC, then quarantined) shipped `cook.pth`, `cook.py` and `sitecustomize.py` into site-packages. The payload was a credential stealer plus an hourly remote loader, and it ran at interpreter start without any import. The Docker `latest` tag carried it too. This was the second compromise this year, after 1.82.7/1.82.8 in March.
- **We were exposed to neither:**
  - The prod container and this Mac had 1.103.0, with no payload files.
  - No `uv.lock` revision held 1.82.7, 1.82.8 or 1.107.0 (the history goes 1.81.9 → 1.82.6 → 1.83.14 → … → 1.103.0).
  - Dependabot PR #102's CI ran 1.104.0.
  - The Docker build installs from the lock with hashes (`--frozen`/`--locked`).
  - Key rotation is not needed.
- Sources: StepSecurity, Wiz, Socket, AISLE, hol.org write-ups on 1.107.0.

## What `033de70` did

**New module `llm/provider.py`.** About 860 lines, stdlib `urllib` only, no SDKs. Interface:
- `chat(model, messages, *, timeout, tools, tool_choice, max_tokens, temperature, effort, json_reply, cache_key) -> Reply`
- `search(model, prompt, *, timeout, cache_key) -> Reply`: xAI Responses API with `web_search`.
- `draw(model, prompt, *, timeout, aspect_ratio, hires, edit_url) -> Picture`: xAI images, generations or edits.
- `Reply(text, tool_calls, finish_reason, usage, cost, id, grounded, thinking)`, with `.turn()` returning the assistant history message (thinking blocks included). Also `ToolCall(id, name, arguments)`, `Usage(input_tokens, output_tokens, cached_tokens)`, `Picture(data, cost, id)`.
- Errors: `ProviderError(message, *, model, provider, status_code, billed)` with subclasses `RequestTimeoutError`, `RateLimitError`, `AuthenticationError`, `NotFoundError`, `BadRequestError` ⊃ {`ContentPolicyViolationError`, `RefusalError`}. They are classified from the status and body in `_http_error`. `billed` means the error body carried usage.
- Internals:
  - Anthropic translation: system messages hoisted, cache breakpoint on the first system block plus top-level `cache_control`, tool and tool_choice mapping, thinking blocks replayed verbatim, `json_schema` sent as `output_config.format`, a 16K default `max_tokens`.
  - xAI: `prompt_cache_key` body field and `x-grok-conv-id` header.
  - `PRICES` (Haiku 5.5 $0.10/$0.50, cache read $0.01, write $0.125; grok-4.3 and both fast aliases $1.25 in, $0.20 cached, $2.50 out, confirmed by owner) and `IMAGE_PRICES`. xAI `cost_in_usd_ticks` (×1e-10, inferred unit) wins when present.
- Tests: `tests/test_provider.py` (57). Request bodies are compared with `tests/fixtures/provider_wire/*.json`, which are litellm 1.103.0's recorded bodies for the same calls; every deliberate divergence is written into the test.

**`service.py` rewired:**
- `_timed_completion` calls `provider.chat(model=…, messages=…, …)` with keywords, so tests can read `call_args.kwargs`.
- `_grounded_completion` calls `provider.search`.
- `_attempt_image_generation` and `_xai_image` call `provider.draw`.
- Error handling is `isinstance` on provider classes; the string-sniffing `_is_provider_refusal`, `_is_content_safety_error` and `_BILLED_FAILURE_MARKERS` are gone.
- New: an Anthropic `finish_reason == "refusal"` (HTTP 200 classifier decline) serves `refusalFallbackModel`, logged with trigger `stop_reason`.

**Deleted, not ported:**
- Gemini paths: safety settings, grounding tools and detection, `_completion_with_tool_fallback`.
- The self-hosted image box and fallback (`rtr.science.uoit.ca:14205`, unreachable from prod and the Mac), and with it config keys `imageApiBase`, `imageSteps`, `imageSize` and `imageFallback{ApiBase,Model,Steps,Size}`.
- `frequency_penalty` (litellm dropped it for both providers), `drop_params`, and `num_retries` (needed `tenacity`, which was not installed, so cleanup would crash on its first error).
- Server-header tracing (`tracing.extract_server_headers`), `save_image_to_http` and `_download_and_save_image`.
- searchModel → assistantModel fallback.
- `ValidatedModelName` now accepts only `anthropic/` and `xai/`, and warns once when a model has no price.
- `apikeys.PROVIDER_ENV_VARS` is now {xai, anthropic}.

**Dependencies:** litellm is gone. The lock went from 104 to 59 packages (openai, pydantic, httpx, tokenizers, huggingface-hub, aiohttp, boto3, …). Dependabot's litellm ignore entry was removed again.

**Bugs found while recording litellm's wire bodies, now fixed:**
- `xai/grok-4-1-fast-reasoning` (prod `codeModel`) had no litellm price, so its spend booked at $0.
- Haiku `json_schema` went out as the deprecated `output_format` with an old beta header.
- Empty assistant turns reached Haiku as the text `[System: Empty message content sanitised to satisfy protocol]`.
- Haiku `temperature` and `frequency_penalty` were silently dropped, so verse sampling never reached Haiku.

## Prod configuration (verified 2026-10-21 from `/home/vibebot/.config/vibebot/bot.conf`)

```
assistantModel: anthropic/claude-haiku-5-5
assistantModel.#CyberCafe: xai/grok-4.3
assistantModel.#HexDroid: xai/grok-4.3
assistantModel.#afternet: xai/grok-4-1-fast-non-reasoning
assistantModel.#tv: xai/grok-4-1-fast-non-reasoning
codeModel: xai/grok-4-1-fast-reasoning
searchModel: xai/grok-4.3
refusalFallbackModel: xai/grok-4-1-fast-non-reasoning
memoryModel / memeModel / verseCompactionModel: anthropic/claude-haiku-5-5
imageModel / memeEditModel: xai/grok-imagine-image
verseModel: (empty) ; verseModel.#afternet: xai/grok-4-1-fast-non-reasoning
subjectResearchModel: (empty) ; animateModel: (empty)
imageFallbackModel: openai//work/models/MiniMax-H3/FL2VA   ← orphan after 033de70
```

**xAI chat is a real chat path, not just the fallback.** Four channels chat on grok, and #afternet runs verse on grok too. #afternet has no refusal fallback, because its model equals `refusalFallbackModel`.

## How model selection works

1. **Route.** `plugin.py` maps each message to a profile (`profile.py`): chat (addressed, `@ask`), code, draw, verse (`@rp` or verse routing), remind_action, and animate (being removed). The profile sets the framework prompt, the visible tools and the output cap. All profiles name `assistantModel`, so `Profile.model_setting` is a constant.
2. **Key per job**, read per channel with `registryValue(key, channel)`:
   - main tool loop: `assistantModel`, or `model_override`;
   - verse: `verseModel`, else `assistantModel`;
   - `@code` and the code tool: `codeModel`, via `completion(command="code")`;
   - search_web and fetch_url: `searchModel`;
   - @draw subject research: `subjectResearchModel`, else `searchModel`;
   - draws: `imageModel`;
   - prompt rewrite, memory cleanup, reminder parse, teasers, status announce (global): `assistantModel`;
   - memory extraction: `memoryModel`, else `assistantModel`;
   - meme picker: `memeModel`, else `assistantModel`;
   - meme pictures: `memeEditModel` (global);
   - verse compaction: `verseCompactionModel`.
3. **Prefix picks the wire** in `provider.py`:
   - `anthropic/` → `api.anthropic.com/v1/messages`, using `ANTHROPIC_API_KEY`;
   - `xai/` → `api.x.ai/v1` (chat/completions, responses, images/*), using `XAI_API_KEY`.
   - There is no load balancing and no failover.
4. **Recovery paths:**
   - Refusal fallback: the whole turn re-runs on `refusalFallbackModel`. It is triggered by refusal-shaped text, `RefusalError`, or `finish_reason == "refusal"`, and is skipped once a non-read-only tool has run.
   - Timeouts on chat, code and draw are stashed in `pending_tasks` and retried on the same model.
   - Auth, bad-request and refusal errors are terminal.
   - xAI prompt-cache key is `chan:<channel>:<lane>` (`_xai_cache_key`). The lanes (main, helper, memory, rewrite, grounded) are live, because grok chat channels run helper and memory ops on grok.

## Owner decisions already made (apply them)

- **Drop:**
  - `effort`: only status announce (global Haiku) uses it. Remove the `effort="low"` arg; keep `max_tokens=800` and the finish_reason check.
  - data-URL images: nothing builds them except animate's `reference_vision_url`.
  - `Reply.id` / `Picture.id`: only an INFO log reads them.
  - The `x-grok-conv-id` header: keep the documented `prompt_cache_key` body field.
- **Remove video/animate entirely** (box gone). Checklist below.
- **Rewrite the operator docs.** They still describe LiteLLM, Gemini, OpenAI and the image box.

## Re-ask the owner (decisions made on a wrong premise)

The owner OK'd dropping these while believing all chat ran on Haiku. It doesn't (see prod config).

- ~~`temperature`~~ **Settled: drop it** (owner, re-confirmed knowing it reaches grok: "we can live with the default temperature"). Remove it from `provider.chat`, `Profile`, the verse profile's 0.8, the assistant loop and the pending-retry stash.
- **`json_reply=True`** (xAI `response_format: json_object`): live for `cleanup_memories` on the four grok channels. memeModel is Haiku, so meme_pick's use is a no-op. Dropping it means relying on the prompt alone for JSON on grok.

## Correctness findings to fix before push (red-team, ranked)

1. **xAI reasoning tokens left out of `output_tokens`** (CONFIRMED). litellm folded `completion_tokens_details.reasoning_tokens` into `completion_tokens`; `_xai_reply` doesn't. Usage rows, `Spend`, `completion_timing` and `@usage` show a drop for grok-4.3 and grok-4-1-fast-reasoning. Fix: `output_tokens = completion + reasoning`, and pass that same value to the cost fallback.
2. **Transport errors escape as raw exceptions** (CONFIRMED by probe). `ConnectionResetError` and `http.client.IncompleteRead` from `getresponse()`/`read()` aren't wrapped, and `exc.read()` on the `HTTPError` path can raise too. Fix: catch `(OSError, http.client.HTTPException)` and map timeouts to `RequestTimeoutError`, everything else to `ProviderError`; read the error body with a try/except fallback to `""`.
3. **POST redirects are followed as GET and carry `x-api-key`/`Authorization` to the new host** (CONFIRMED by probe). Unlikely with fixed https hosts, but fix it: `build_opener` with a `HTTPRedirectHandler` subclass whose `redirect_request` raises.
4. **Refusal vs block precedence** (PLAUSIBLE). `_http_error` checks image-moderation markers before refusal markers, so a chat refusal body containing "content policy" becomes `ContentPolicyViolationError`: no fallback, and the "violates AI safety" line. Fix: check `_REFUSED_MARKERS` first.
5. **Unverified request details** (settle with the smoke test). There is no explicit `User-Agent` (urllib sends `Python-urllib/3.14`), so set `User-Agent: vibebot`. `draw()` sends `n: 1` on `/images/edits`; the old `_xai_image` didn't, so drop `n` for edits.
6. *Pre-existing, optional:* a draw that is charged but returns no picture books $0, because `_attempt_image_generation` returns None and drops `picture.cost`.
7. *Accepted behaviour change:* a 403 without refusal prose is now `AuthenticationError` (terminal, "Invalid API key"), where litellm made it a generic, retried error.
8. *Accepted:* no connection reuse (a new TLS handshake per call, ~100–300 ms).

## YAGNI cuts (red-team). Do the certain ones; discuss the rest.

**Certain / near-certain:**
- `completion()` (service.py ~3642–3806) has one caller, `_code_for_assistant`, which passes `prompt`, `command="code"` and `system_prompt`. These are dead: `images`, `history`, `channel_history`, `irc`, `msg`, `memories`, `user_instruction`, `model_override`, `_filter_images`, `_build_messages`, the stash branch, and `CompletionResult.grounding_used` (always False). Shrink to about 30 lines.
- `_anthropic_tool_choice` `"auto"` entry, `_ANTHROPIC_FINISH` `stop_sequence`/`pause_turn`, the `_tool_input` dict branch, `IMAGE_PRICES["xai/grok-imagine-image-quality"]`, and the `_billed` non-JSON branch (the repr form came from litellm's wrapper).
- Error family: drop `NotFoundError` and the special 408/413/422 mappings. Rule: refused → `RefusalError`, blocked → `ContentPolicyViolationError`, 429 → rate limit, 401/403 → auth, other 4xx → `BadRequestError`, else `ProviderError`. This also stops a stray 422 being retried. Drop `ProviderError.model`/`.provider`/`.status_code`/`.message`; `_xai_image`'s `refused = status in (400, 403, 422)` becomes `isinstance(exc, BadRequestError)`.
- Moderation markers: only `content-moderated`/`content moderation` (xAI images) and `violates usage guidelines`/`content violates` (xAI chat 403) are confirmed for our providers. `moderation_blocked`, `safety system` and `content policy` are OpenAI-era, `safety filter` is Gemini-era, and `safety_check` and `usage policies` are unconfirmed. **Keep `RefusalError`**: without it a grok 403 refusal reads as "Invalid API key".
- `_grounded_completion` hand-builds its `completion_timing` lines; reuse `_log_completion_timing`. Its INFO start/ok lines never reach prod. `_ask_completion` logs failures at INFO, so they are invisible in prod; make that WARNING.

**Judgment calls (discuss):**
- The 12 `_missing_key_error` pre-checks duplicate `provider._key` raising `AuthenticationError`. Cutting them loses the "(set X_API_KEY)" wording and makes the fallback-gate check lazy.
- The duplicate unpriced warning: config validation already warns at load. The provider one only adds coverage for models reached through persisted pending rows.
- `subjectResearchModel` is unset in prod; drop the key and the `model=` parameter on `_grounded_completion`.
- Ticks vs price table for xAI chat and search: not established that `/chat/completions` and `/responses` return `cost_in_usd_ticks`. Settle it with the smoke test, then keep one path.
- `Profile.model_setting` is the same key on all profiles.
- The `verseModel`-empty warning (plugin.py ~8046) names gemini-flash-latest and fires for every verse channel on Haiku.
- Stale comments citing Gemini: status announce (~4464), memeModel description (config ~1178), plugin.py ~6234 ("gemini and grok"), apikeys "Gemini 39".

**Not cuts:**
- `cache_write` pricing: live, since every Haiku cache miss writes.
- `DEFAULT_MAX_TOKENS`: Anthropic requires it.
- The thinking pass-through: Haiku 5.5 thinks by default.
- The xAI cache lanes: live on the grok chat channels.
- The `RefusalError` fallback branch: live on #CyberCafe and #HexDroid.
- The grok workarounds (`<|eos|>` strip, non-dict tool args, `content or ""`, echo/verse-denial/safety-refusal guards): live on the grok channels and the fallback.

## Video/animate removal checklist (red-team, high confidence)

Paths are under `plugins/llm/src/llm/`.

**Remove:**
- **config.py**
  - The 14 `animate*` keys (~291–455).
  - The animate rate-limit block (~1057–1066), which generates `animate{,Trusted,Unreg}RateLimit{Count,Window}` by f-string.
  - `animateExpiry` is read by f-string (`f"{task_type}Expiry"`, service.py ~3132).
  - The litellm/vLLM comment at ~293.
- **apikeys.py:** `ANIMATE_ENV_VAR` and `animate_api_key()`.
- **profile.py:** `PROFILE_ANIMATE` and its profile.
- **prompts.py:** `ANIMATE_SYSTEM_PROMPT`, `ANIMATE_REFERENCE_OVERLAY`, and the `"animate"` PROMPTS entry.
- **assistant.py:** the `generate_video` schema and policy, `animate_fn`, `_tool_generate_video`.
- **service.py**
  - `EXPLICIT_VIDEO_RE`, `ANIMATE_POLL_INTERVAL_SECONDS`.
  - `split_reference_url` (animate-only callers).
  - The video-fabrication guard (`_ANY_URL_RE`, `_VIDEO_FABRICATION_*`, `_strip_urls`).
  - `ReferenceImage`, `VideoResult`.
  - Poll and dispatch branches (~3245–3265, ~3523), `force_initial_video` (~5405–5514), ~5630–5650, ~5851.
  - The whole video section (~6291–6870: `_animate_*`, `animate_available`, `video_available`, `reference_vision_url`, `fetch_reference_image`, `video_generation`, `_retry_video`, `cancel_video`, `_save_video_bytes`).
  - The mp4 MIME and upload branches (~7154, ~7258).
- **plugin.py**
  - `_ANIMATE_DEFAULT_MOTION`, the `llm.animate` capability, and the animate, `video` and renders command table (~426–444).
  - The render-typing thread and `_animate_admit_lock` (~1048–1068), and their teardown in `die`.
  - `_RENDER_*`, the animate delivery formatting (~2780–2787, ~4573–4610), `_render_typing_loop` and `_typing_refresh_pass`.
  - `_animate_for_assistant`, `_submit_video`, `_animate_admission`, the `animate_fn` wiring.
  - The `animate` command, `_pending_animate_rows`, `_cancel_animate_row`, `@renders`, `_animate_reference_for`, and `video = animate`.
- **persistence.py:** `active_animate_targets`, `count_pending_animate`, `count_pending_animate_for`.
- **typing_holds.py:** `set_group`/`_groups` become dead (only the render refresher uses them). Keep `hold()`.
- **Tests to delete:** test_animate.py, test_animate_reference.py, test_animate_reference_command.py, test_renders_command.py.
- **Tests to edit:** conftest (33 hits), test_persistence, test_feature_switches, test_subject_research_wiring, test_profile, test_plugin_delivery, test_prompts, and single hits in test_caller, test_integration, test_commands, test_plugin_dispatch, test_plugin_core, test_subject_dossier.
- **Docs:**
  - `.env.example` animate block, README:13, `mkdocs.yml:34`.
  - Guide pages: `docs/guide/plugins/llm.md`, `user/ai-commands.md`, `operator/{configuration,tuning-monitoring,rate-limiting-security}.md`.
  - Formal model: `docs/formal/AnimateAdmission.tla`, `MCAdmissionTRUE/FALSE.tla`, `docs/formal/check.sh:27–28`, `docs/formal/README.md:12–13`.

**Keep. Shared with live features:**
- `pending_tasks` and its stash/expiry machinery (ask, code, draw).
- `TypingHolds.hold()`.
- Subject research for @draw. Only reword "image or video" in `subjectResearchEnabled` and the dossier prompt.
- `_format_duration` (used by uptime).
- The `_JOB_MARKER_RE` guard: it also catches `[Generated image: …]`. Rewrite its nudge, which tells the model to call `generate_video`, instead of deleting the guard.
- `usage.rendered_prompt` (migration v18): only animate writes it. Leave the column; dropping it needs a migration.

**Prod after removal** (edit bot.conf only while the bot is stopped):
- The `supybot.plugins.LLM.animate*` lines and `imageFallback*`/`imageApiBase` lines become orphans and can be deleted.
- `ANIMATE_API_KEY` in the env file can go.
- Delete leftover `pending_tasks` rows with `task_type='animate'`.

## Docs rewrite scope

These still describe LiteLLM, Gemini, OpenAI, Vertex and the image box: `README.md`, `AGENTS.md` (lines 7, 50, 72, 77), `.env.example`, `docs/guide/index.md`, `docs/guide/operator/configuration.md` (provider keys, local servers, Vertex, self-hosted endpoint, fallback, @animate, "LiteLLM's provider/model format"), and `plugins/llm/README.md`. The house style contract is in memory `project_docs_revamp_2026_07_09`: `@` prefix, en-CA, sentence case; Vale is local only. CHANGELOG is generated by git-cliff, so don't edit it.

## Before pushing

1. Apply the owner decisions, correctness fixes 1–5, and the certain YAGNI cuts. Get `make check` green.
2. **Smoke test against the real APIs, with the owner's OK.** It spends cents and uses prod keys. One request each: Anthropic messages (with a tool and a cached system prompt), xAI chat completions, xAI responses + web_search, xAI images/generations, xAI images/edits. Confirm: 200s, the `User-Agent` is accepted, `cost_in_usd_ticks` presence on chat and responses, and `cache_read_input_tokens` > 0 on a repeated Haiku call.
   - Keys exist only in prod's env file (`/home/vibebot/.config/vibebot/env`).
   - Prod has 961 MB of RAM: run a stdlib-only script, not a plugin import.
3. Video removal as its own commit; docs as their own commit.
4. Push. Then watch `completion_timing` (`result=error`), `refusal_fallback_served` and image cost rows in prod logs for an hour.

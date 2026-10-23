# AGENTS.md: instructions for coding agents

This file is the canonical repository guidance for coding agents. `CLAUDE.md` points here.

## Project summary

**VibeBot v8** is a Limnoria-based IRC bot workspace with AI features on Anthropic and xAI models.

Workspace members:

- `plugins/llm/`: the main LLM plugin (assistant, verse, scheduler, persistence)
- `plugins/nickinmiddle/`: companion `inFilter` plugin that moves a mid-message bot nick to the front, so Limnoria's normal addressing logic fires

Core documentation and tools:

- `README.md`: project overview and operator quick start
- `docs/guide/`: MkDocs user and operator guide (published to GitHub Pages)
- `Makefile`: standard development entry points
- `pyproject.toml`: workspace, Ruff, pytest, coverage, and ty configuration

The bot's command prefix on AfterNet is `@` (for example `@ask`). The `%` prefix in a local development `bot.conf` (gitignored, not in the repo) is a development artifact; docs use `@`.

## Working in this repo

- Prefer `make` targets over raw tool commands when a matching target exists.
- After editing Python files, run `make lint` and `make typecheck`.
- Before considering work complete, run `make preflight`.
- If `make preflight` is too expensive for a narrow change, run the most specific checks that cover the edited area and state what you did not run.
- Use `make test` for the normal suite and `make test-all` when slow tests are relevant.
- Run `make docs` when changing MkDocs content or navigation, and `vale <files>` on changed prose.

Key targets:

```bash
make install
make run
make lint
make typecheck
make test
make preflight
make docs
```

## Architecture

For `plugins/llm/src/llm/`:

- `plugin.py`: IRC protocol layer, command routing, and command wrappers
- `service.py`: business logic: completions, tool loop, sanitisation, output shaping
- `provider.py`: the only code that talks to model providers: stdlib HTTP to Anthropic (`anthropic/`) and xAI (`xai/`), wire formats, prices, error mapping
- `assistant.py`: tool-using chat profile, tool schemas, and tool wrappers
- `profile.py`: route profiles (`chat`, `verse`, `draw`, `remind_action`) that decide which tools an entry point advertises
- `prompts.py`: the four framework system prompts, the memory-pipeline prompts, and the shared `IRC_OUTPUT_FORMAT` block
- `executor.py`: `LLMExecutor` (`BoundedSemaphore` + `ThreadPoolExecutor`); every blocking LLM call goes through `permit()` or `submit()`
- `persistence.py`: SQLite store for memories, reminders, scheduled tasks, and usage; uses the `_write_txn` context manager for atomic writes
- `limnoria_bridge.py`: allowlisted Limnoria-as-tool surface (mutation gating plus a curated default allowlist)
- `config.py`: Limnoria registry configuration (models, prompt overlays, context, memory, rate limits, verse, bridge, status page). No API keys — those come from the environment; see `apikeys.py`
- `context.py`: conversation history and thread-safe state
- `identity.py`: `Identity` and `Caller` value types (storage key, personal key, reply target); resolved once per message by `LLM._resolve_caller`
- `apikeys.py`: provider-scoped key resolution (model → provider → environment variable) and the `SecretFilter` that scrubs key values out of every log handler
- `agenda.py`: the agenda — when reminders and scheduled LLM tasks fire: the one-minute cadence floor, next-fire anchoring, per-kind chain caps, the row-is-the-cancel-truth check, restart recovery, and the next due time the plugin's single wakeup timer is armed for; free of Limnoria imports, tested against real SQLite and a fake clock
- `statuspage.py`: Atlassian Statuspage v2 model — fetch, strict parse, incident classification, sanitisation, line rendering; deliberately free of Limnoria and `service` imports
- `tracing.py`: per-request trace ids (`request_id` ContextVar, `generate_request_id`, and `TraceFilter`, which prefixes log lines with the id)
- `verse/`: the verse subsystem (SQLite world store, avatars, aging, compaction, reactions, taste command-line tools, validation); the two-step `@versepurge` lives in `plugin.py`

Keep those boundaries intact:

- IRC command parsing and reply flow belong in `plugin.py`.
- Provider wire calls belong in `provider.py`; sanitization and output shaping belong in `service.py`.
- Registry options belong in `config.py`.
- Shared mutable state must remain thread-safe because `Plugin.threaded = True`.
- All blocking LLM calls must go through `LLMExecutor` (no direct `provider.*` call from the IRC main thread).
- Use `%i`, never `%d`, in log format strings. Supybot installs its own logger class process-wide, and its formatter has no `%d`: the literal stays in the line and the remaining arguments shift into the wrong slots, with no error. `test_log_format_specifiers.py` fails on any `%d` in a logging call.

## Security invariants

- Never log, echo, or persist API keys in plain text.
- API keys come from the environment, one variable per provider (`ANTHROPIC_API_KEY`, `XAI_API_KEY`), resolved in `apikeys.py`. Do not move key handling into the Limnoria registry — the `LLM.*ApiKey` settings were deliberately removed.
- Scrub secrets from user-visible error messages.
- Treat channel topics, stored memory, reminders, and other recovered content as untrusted input.
- Check image and external URLs; block unsafe schemes such as `javascript:`, `data:`, and `file:`.
- Prevent path traversal with resolved-path checks.
- Preserve IRC command-injection defences for generated output.
- Keep HTML sanitization in place for rendered output.

## Testing expectations

- Add or update tests for behaviour changes.
- Before writing or reviewing tests, read `.claude/skills/writing-tests/SKILL.md`. It covers where expected values come from, what to assert, which fakes to use, and the scoped mutation check that proves a new test kills something.
- LLM plugin tests live under `plugins/llm/tests/`; nickinmiddle tests under `plugins/nickinmiddle/tests/`.
- Keep the **93%** coverage floor (enforced by `pyproject.toml` and `make test`).
- Follow existing test style, including behaviour-driven docstrings where that pattern already exists.
- Property-based tests (Hypothesis) live as `test_*_properties.py` alongside example tests; prefer extending a property test over adding a new example when the invariant generalizes.
- Include security and concurrency coverage when touching those areas.

## Common tasks

### Adding or changing IRC commands

1. Update `config.py` if the command needs new settings.
2. Add or change the command wrapper in `plugin.py`.
3. Adjust business logic in `service.py` if applicable.
4. Add or update tests.
5. Update `docs/guide/` if user-facing behaviour changes.

### Changing LLM behaviour

- Framework prompts live in `prompts.py` (see Architecture); verse prompt assembly is `build_verse_system_prompt` in `verse/avatar.py`. Only `assistantSystemPrompt` is a channel-overridable registry setting.
- Model selection is per-purpose. The registry holds `assistantModel`, `memoryModel`, and `verseModel`; image, search, meme, refusal-fallback and verse-compaction models are fixed constants (`IMAGE_MODEL`, `SEARCH_MODEL`, `MEME_MODEL`, `MEME_EDIT_MODEL`, `REFUSAL_FALLBACK_MODEL` in `service.py`; `_VERSE_COMPACTION_MODEL` in `plugin.py`). Bumping the chat model does not move verse; update both when that is the intent.
- API keys are resolved from the environment at call time, keyed on the provider of the model being called (`apikeys.api_key_for`). A model with any prefix but `anthropic/` or `xai/` is rejected at `@config` time.

## Important files

- `plugins/llm/src/llm/service.py`: main LLM execution path
- `plugins/llm/src/llm/plugin.py`: command surface and IRC integration
- `plugins/llm/src/llm/context.py`: volatile memory implementation
- `plugins/llm/src/llm/limnoria_bridge.py`: Limnoria-to-LLM tool bridge
- `plugins/llm/src/llm/agenda.py`: the single source of reminder and scheduled-task timing; change cadence, cap, or restart rules here, not in `plugin.py` or `service.py`
- `plugins/llm/src/llm/{service,plugin,assistant,persistence}.py`: native `schedule_llm_task` plus the unified `list_pending_tasks` / `cancel_pending_task` / `cancel_all_pending_tasks` tool surface
- `plugins/llm/src/llm/executor.py`: global LLM concurrency gate; the cap is `_MAX_CONCURRENT_LLM_CALLS` in `plugin.py`
- `plugins/nickinmiddle/src/nickinmiddle/`: nick-in-middle `inFilter` plugin
- `mkdocs.yml` and `docs/guide/`: published guide source
- `docs/plans/`: design and implementation plans; shipped plans move to `docs/plans/archive/`. Check the active list in [`docs/plans/README.md`](docs/plans/README.md) before starting work on a subsystem — it is where open follow-ups that are waiting on production traffic are parked, and the measurement they are waiting for is easy to do accidentally and then lose.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `rdrake/vibebot`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels are used as-is: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root (`docs/adr/` is created when the first decision is recorded). See `docs/agents/domain.md`.

## Go rewrite (v9)

The Go rewrite lives in `go/` with its own Makefile: build and test with `cd go && make all`. Specs and plans live under `docs/superpowers/`. The Python tree in `plugins/` remains the production bot.

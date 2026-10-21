# LLM plugin for Limnoria

The main AI plugin for VibeBot v8: multi-provider chat, code and image generation, memory, reminders, scheduled tasks, and the verse fiction layer, all through LiteLLM.

This README covers plugin internals for developers. User and operator documentation lives in the [published guide](https://rdrake.github.io/vibebot-v8/).

## Module layout

```
src/llm/
├── plugin.py           # IRC protocol layer: command wrappers, routing, doPrivmsg/doTagmsg
├── service.py          # LiteLLM calls, sanitization, output shaping, storybook generation
├── assistant.py        # Tool-using chat profile and tool specs
├── executor.py         # LLMExecutor: bounded concurrency for all blocking LLM calls
├── persistence.py      # SQLite store: memories, reminders, scheduled tasks, usage
├── limnoria_bridge.py  # Allowlisted Limnoria-as-tool surface
├── config.py           # Registry configuration (supybot.plugins.LLM.*)
├── context.py          # Volatile conversation history, thread-safe
├── tracing.py          # Structured trace severity helpers
└── verse/              # Verse subsystem
    ├── store.py        # SQLite world store (entities, events, relations, aliases)
    ├── avatar.py       # Verse prompt assembly, tool specs, verse_edit dispatch
    ├── aging.py        # Auto-created entity retirement sweep
    ├── compaction.py   # Daily retention job: old events become a lore digest
    ├── reactions.py    # IRCv3 reaction capture and reporting
    ├── taste_mine.py   # Offline CLI: mine style exemplar candidates from logs
    ├── taste_report.py # Offline CLI: verse landing-rate report
    ├── validation.py   # Payload validation for verse mutations
    └── purge.py        # Two-step verse wipe
```

Boundaries to respect:

- IRC parsing and reply flow stay in `plugin.py`; provider calls and output shaping stay in `service.py`.
- Every blocking LLM call goes through `LLMExecutor` (`permit()` or `submit()`); never call `litellm.*` from the IRC main thread.
- Shared state must stay thread-safe: `Plugin.threaded = True`.

## Commands, capabilities, and gates

| Command | Capability | Notes |
|---------|------------|-------|
| `@ask` | `llm.ask` | Context, vision, tool loop |
| `@draw` | `llm.draw` | Requires an authenticated account |
| `@story` | `llm.draw` | Illustrated story or explainer page; counts against the draw limits |
| `@forget`, `@memories`, `@instruct`, `@remind`, `@usage`, `@avatar` | none | Owner-only subcommands enforced in-body |
| `@verseopt`, `@verse`, `@look`, `@who` | `llm.verse` | Verse participation |
| `@canon`, `@versedit` | `llm.verse.edit` | Canon editing |
| `@versedump`, `@versepurge`, `@versecompact` | `llm.verse.gm` | GM operations; `versedump`/`versepurge` check the capability in-body |

Rate limiting covers four command families (`ask`, `draw`, `meme`, `story`), each with registered, trusted, and unregistered tiers. The [operator guide](https://rdrake.github.io/vibebot-v8/operator/rate-limiting-security/) documents the matrix and defaults.

## Security patterns

1. **Thread-safe API keys**: keys pass directly to `litellm.completion()`; the plugin never mutates environment variables, which prevents races between threads.
2. **API key sanitization**: `_sanitize()` strips keys from every error message before logging.
3. **URL validation**: `validate_image_url()` blocks non-HTTP schemes (`javascript:`, `data:`, `file:`, `ftp:`), path traversal, and non-image extensions.
4. **Private configuration**: every API key registry value is `private=True`.
5. **Output defences**: generated output is sanitized against IRC command injection, and rendered HTML passes through nh3.

## Testing

```bash
make test      # from the repo root; skips slow tests, enforces 93% coverage
make test-all  # includes slow tests
```

Tests live in `plugins/llm/tests/`. Property-based tests (Hypothesis) sit alongside example tests as `test_*_properties.py`; extend a property test when the invariant generalizes.

## Adding a command

1. Add the command method to `plugin.py`:

```python
def mycommand(self, irc, msg, args, text):
    """<args>

    Help text here.
    """
    # Skip ZNC playback messages
    if self._is_old_message(msg):
        return

    # Your logic here
    irc.reply("Response")


mycommand = wrap(mycommand, ["text"])
```

2. Register any new settings in `config.py`.
3. Guard expensive commands with a capability (`checkCapability` in the `wrap` spec) and consider a rate-limit family.
4. Add tests in `tests/`, then run `make preflight`.

## Code style

- Ruff for linting and formatting; ty for type checking
- Type hints on all functions; docstrings on all public methods

## Licence

See the LICENSE file in the repository root.

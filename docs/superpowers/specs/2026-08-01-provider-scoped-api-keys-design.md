# Provider-scoped API keys — design

**Status:** approved, red-teamed twice (spec and plan)
**Date:** 2026-08-01

## Problem

API keys are configured per *role* and per *channel*, so one credential is
duplicated across many registry entries and drifts out of sync. Prod `bot.conf`
holds twelve `apiKey` entries covering four distinct key values:

| Entry | Key | Provider |
|---|---|---|
| `assistantApiKey` (global) | KEY-A | gemini |
| `searchApiKey` | KEY-A | gemini |
| `assistantApiKey.#afternet` / `.#CyberCafe` / `.#HexDroid` / `.#tv` | KEY-B | xai |
| `assistantApiKey.\:afternet.#afternet` / `.#cybercafe` / `.#hexdroid` | KEY-B | xai |
| `assistantApiKey.\:afternet.#tv` | KEY-C | xai |
| `codeApiKey` | KEY-D | xai |
| `imageApiKey` | KEY-B | xai |

The three xAI keys (B, C, D) are **unintentionally** different — a symptom of the
management problem, not a design choice. `#tv` resolves to KEY-C through its
network-scoped entry while its plain-channel entry says KEY-B.

The original reason keys were scoped per channel — a mix of free and paid Gemini
keys — **no longer applies**. There is one Gemini key in use, the paid one
(project `417371945641` / `gemini-api-paid-469818`), and the `vibebot-free` key
is not referenced anywhere in `bot.conf`. Grounded search is free on the paid key
(~336 requests/month against a 5,000/month allowance).

Because keys and models are configured independently, they have drifted apart.
Three live cross-provider mismatches exist today, all verified in code:

1. `service.py:2631-2637` — the pending-task retry path stores `task_type="code"`
   with `model=assistantModel` (the code profile's `model_setting` is
   `assistantModel`, `profile.py:105`), then pairs it with **`codeApiKey`**. In
   prod that is KEY-D (xai) sent with whatever `assistantModel` resolves to.
2. `service.py:3179-3184` — grounded search resolves its model as
   `searchModel or assistantModel` and its key as `searchApiKey or
   assistantApiKey`, independently. `searchModel` defaults to `""` and
   `searchApiKey` has no channel override, so on grok channels a Gemini key
   (KEY-A) is sent with an xAI model.
3. `plugin.py:6347-6348` — verse compaction pairs `registryValue("assistantApiKey")`
   with a `gemini/` model (registry value with a `gemini/` fallback).

## Goal

A key is a property of **which provider you are paying**, not of which channel or
which role. One key per provider, set once in the container environment, resolved
from the model actually being called — which makes the three mismatches above
structurally impossible. The code that existed only to reconcile per-channel and
per-role keys is deleted along with the entries.

## Approach

Key selection becomes a pure function of the model:

```python
api_key_for("xai/grok-4.3")  ->  os.environ["XAI_API_KEY"]
```

Three decisions give that its shape.

**Resolve at the outbound boundary, not at the caller.** Every request to a
provider passes through exactly four functions — `_timed_completion`
(`service.py:2351`), `_xai_responses_call` (`3281`), `_generate_image_once`
(`4150`), and `LiteLLMVerseClient.call` (`verse/compaction.py:278`). Resolving
there rather than at the ~14 places that currently read a key deletes most of the
plumbing outright and makes the policy impossible to bypass: a future caller
cannot pass the wrong key because it cannot pass a key at all. It also removes a
class of bug the caller-side approach kept — `service.py:3036` computes the
outbound model as `model_override or registryValue(...)`, so a guard resolving
the registry value alone would still mismatch whenever an override is in play.

**Derive the provider from the model via LiteLLM.** `provider_of` wraps
`litellm.get_llm_provider(model)[1]` rather than splitting on `/`. Unprefixed
model names are legal — `config.py:40` validates models through litellm, which
resolves `gpt-4` and `dall-e-3` to openai (verified; note it *raises* on bare
`claude-3-opus`, so the wrapper must catch) — and the test suite uses unprefixed
names in roughly 100 places.

**Map what we pay for; delegate the rest.** Four providers get an environment
variable and an explicit missing-key error. Any other provider LiteLLM
recognises — `vertex_ai`, `openrouter`, `azure`, `bedrock` — resolves to `None`,
which is exactly what makes LiteLLM consult its own native credential mechanism
(ADC, `OPENROUTER_API_KEY`, IAM roles). This is not a gap; it is what keeps the
plugin multi-provider. `ValidatedModelName` accepts anything LiteLLM accepts, and
a four-entry allowlist that hard-errored on everything else would narrow a
general plugin to four providers and break the shipped `vertex_ai` image default.

The four registry settings are deleted outright, not retained as an override.
Retaining them would preserve every mechanism this change exists to remove.

## Components

### New module: `plugins/llm/src/llm/apikeys.py`

`service.py` is already oversized, and key resolution depends on nothing in the
service, the plugin, or Limnoria. It goes in its own module, testable as a pure
function.

```python
PROVIDER_ENV_VARS = {
    "xai": "XAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

def provider_of(model: str) -> str          # "" when LiteLLM cannot classify it
def env_var_for(model: str) -> str | None   # variable NAME, for error messages
def api_key_for(model: str) -> str | None   # variable VALUE
def is_managed(model: str) -> bool          # provider is in PROVIDER_ENV_VARS
```

`provider_of` must never raise: LiteLLM raises `BadRequestError` on names it
cannot place, and every caller sits on a path with its own error handling.

`api_key_for` reads the environment on every call rather than caching at import,
so the value redaction scrubs cannot diverge from the value actually sent.

### Missing-key policy

Exactly one rule, applied at the four boundaries:

- `is_managed(model)` and the variable is empty → configuration error naming the
  provider and the variable: `"no API key configured for provider 'xai' (set
  XAI_API_KEY)"`. A bare "API key not configured" sends an operator hunting a key
  that is set when the real problem is the model's provider.
- Otherwise → pass `None` and let LiteLLM resolve natively.

The eight guards that currently report a missing key (`2638`, `2718`, `3029`,
`3602`, `3788`, `4091`, `4328`, `5117`) keep their user-visible error paths, but
their *condition* moves to the boundary. Callers that guard purely to produce a
friendly message keep doing so by asking `is_managed` and `api_key_for`; they no
longer choose or forward a key.

### Redaction

`_sanitize` (`service.py:1222`) is applied by hand at ~25 call sites and misses
the path that leaks most. Supybot's `Logger.exception` (`supybot/log.py:77-85`)
writes the raw traceback *and* calls `utils.python.collect_extra_debug_data()`,
which `repr()`s every local in every frame plus every attribute of `self`.
`verse/compaction.py` and `executor.py` have no `_sanitize` coverage at all, and
`compaction.py:276` stores the key on `self` where the attribute walk finds it.

Add a `logging.Filter` that scrubs `record.msg`, `record.args`, `record.exc_text`
and `record.stack_info`, sourcing values from the environment.

**Install it on handlers, not loggers.** A logger's filters run only for records
originating on that logger; propagation to an ancestor runs the ancestor's
*handlers*, not its *filters*. The plugin logs through at least ten loggers
across two hierarchies — `supybot.plugins.LLM{,.service,.assistant,.config,.bridge}`
and `llm.verse.{compaction,store,purge,avatar,aging}` — so attaching to two
loggers would cover two. Handler-level installation covers every record that
reaches an output sink regardless of origin. Handlers added later (supybot
creates per-plugin file handlers when `individualLogfiles` is true; prod has it
false) are not covered, which is a documented limitation, not a silent one.

Scrubbing must never break logging: a lone `Mapping` argument that is not a
`dict` must not be iterated as a tuple, or `getMessage()` raises
`TypeError: format requires a mapping`. Arguments are scrubbed by their `str()`
value, not by `isinstance(arg, str)` — provider `AuthenticationError` objects
echo the submitted key and are routinely logged as `log.error("...: %s", exc)`.
(Note that supybot's own `Logger._log` pre-formats args into `msg`, so the args
branch is dead for supybot loggers and live for the `llm.verse.*` ones.)

`_sanitize` is **kept**, sourcing from the same helper, because two paths are not
logging: `service.py:5171-5239` feeds `self._sanitize(str(e))[:200]` back into
`_rewrite_prompt` as **prompt text** (LiteLLM places Gemini keys in the request
URL query string, so an unsanitized error here could be laundered through the
model into channel output), and `AssistantResult.error` is persisted by
`db.log_usage`.

Value-replacement redaction is defence in depth, not a boundary: it cannot catch
an encoded, truncated, or transformed credential.

### Deletions

These exist only to reconcile per-role and per-channel keys:

| What | Where |
|---|---|
| Four `registerChannelValue` key blocks | `config.py:131-171` |
| `configure()` wizard key line | `config.py:120` |
| `api_key_setting` field, `"assistantApiKey"` on all five profiles | `profile.py:60,80,93,106,118,132,157` |
| `api_key_name` ladder (whole block) | `service.py:2631-2636` |
| `api_key_name` lines **only** (`3021`, `3025`) | `service.py:3020-3026` |
| `api_key` parameters and their `or` lines | `service.py:2969/3028`, `3500/3544`, `4242/4325` |
| `api_key` parameter threading through `_completion_with_tool_fallback` and `_xai_responses_call` | `service.py:1866`, `3281` |
| `channel` parameter of `_generate_image_once` | `service.py:4150,4158,4175` |
| `api_key` parameter of `LiteLLMVerseClient` | `verse/compaction.py:273,276,285-286` |
| Stale docstring and comment references | `service.py:2995`, `profile.py:101-102` |

`service.py:3020-3026` must **not** be deleted as a block: the same `if/else`
selects `codeModel`/`codeSystemPrompt` versus `assistantModel`/
`assistantSystemPrompt`. Only the two `api_key_name` lines come out.

The three caller-facing `api_key` parameters are dead in `src/` **and** in tests:
no caller in `plugins/llm` passes one, the `**` spreads at those call sites
expand `_pending_task_fns` (callables only), and there is no `functools.partial`
or dynamic dispatch reaching them. `model_override` alongside them **is** live
(verse, storybook) and stays. `_generate_image_once`'s `channel` parameter exists
solely for the per-channel `imageApiKey` lookup and also passes a raw
`msg.args[0]` — a nick in a PM — as a registry scope, unlike every other site.

**`imageModel`'s `vertex_ai` default stays.** Under the delegate-the-rest rule it
resolves to `None` and LiteLLM uses ADC, so there is nothing to fix — and
critically, nothing to migrate on prod, which almost certainly carries an
explicit `imageModel` value that a default change would not have touched anyway.

Documentation that becomes wrong:

| What | Where |
|---|---|
| Env-var guidance contradicting this design | `.env.example` (copied to the live env path by `Makefile:243-249`) |
| API-key table and `@config` example | `docs/guide/operator/configuration.md:25-28,33` |
| Key references, incl. the troubleshooting step "`@config plugins.LLM.assistantApiKey` shows a masked value" | `docs/guide/operator/tuning-monitoring.md:22,25,166` |
| Key reference | `docs/guide/operator/memory-promotion.md:88` |
| Key configuration example, and the claim that keys live in the registry | `README.md:34,55` |
| Rollback procedure (absent today) | `docs/guide/operator/operations.md` |
| New environment exposure (see below) | `docs/guide/operator/operations.md` |

`tuning-monitoring.md:166` matters disproportionately: it is the diagnostic an
operator would reach for *during* this migration, and it stops working.

Locale files contain no `ApiKey` msgids, but deleting `_()` docstrings will churn
`messages.pot` on next extraction.

## Testing

The suite collects 2572 and runs 2558 (`make test` excludes 14 `slow` tests),
with `fail_under = 93` (`pyproject.toml:75`) against a current 94%. Measured:
`plugins/*/src` is 7920 statements, so even deleting 300 fully-covered ones lands
near 93.8% — **the coverage gate is not a hazard at any intermediate commit.**
What blocks a commit is test failure.

**Environment isolation comes first, before any other test is written.**
`litellm/__init__.py:27` calls `load_dotenv()` at import and `.env` is gitignored,
so importing `llm.service` injects a developer's **real** keys into `os.environ`
before collection. Until an autouse fixture scrubs every secret-suffixed variable
and sets four fakes, a test asserting on the collected secret set will render
real keys into a pytest failure diff. The suite also has no network guard, and
handing every test plausible credentials turns any incompletely-mocked path into
a live outbound request — so a socket block belongs in the same fixture.

Verified: no file under `plugins/llm/src/` or `plugins/llm/tests/` reads or
writes `os.environ` today, so the fixture changes no existing behaviour.

Test migration is larger than a naive grep suggests, in two directions. Guards
that **stop** firing: `test_provider_edge_cases.py:491,499`,
`test_service_images.py:971,1145,1318`, `test_service_completion.py:1587`,
`test_service_memory.py:49,63`. Guards that **start** firing were the surprise —
under a hard-error rule, the 14 tests in `test_service_images.py`'s
`TestDrawAutoRewrite` (`vertex_ai/...`) and `test_etiquette.py:119` (bare
`"imagen"`) would all have broken. The delegate-the-rest rule removes that class
entirely.

`test_stress.py` is `pytestmark = pytest.mark.slow` and never runs under
`make test` — a test that "passes" there proves nothing. Any verification step
relying on it must invoke it explicitly.

## Migration

Verified against Limnoria's registry implementation:

- Unknown keys in `bot.conf` are read into a flat cache and never validated
  (`supybot/registry.py:78-113`), so a `bot.conf` holding entries for
  now-unregistered settings loads without error. The repo's own local `bot.conf`
  proves it — it still carries `askApiKey` and `drawApiKey`, renamed away long
  ago. The deploy and the `bot.conf` cleanup can therefore be separate steps.
- `registry.close` writes only the *registered* tree (`registry.py:130-177`), and
  `world.upkeep` flushes on a timer — `supybot.flush` defaults `True`
  (`conf.py:795`), `supybot.upkeepInterval` defaults 3600 (`conf.py:789`). So
  **within about an hour of the deploy, `bot.conf` rewrites itself without the
  twelve entries, with no operator action.** The cleanup step is largely
  automatic, and the rollback material disappears on a timer.

**Preconditions, before anything changes:**

- Set `supybot.commands.allowShell: False` (currently `True`, `bot.conf:153`).
  The `Debug` plugin's `environ` command replies `repr(os.environ)` **to the
  channel** and is one `@load` away for an owner. Before this change the worst
  owner fat-finger was one key delivered to a PM by `@config`; afterwards it is
  all four keys in a channel at once. This is a precondition, not a follow-up.
- Confirm `supybot.flush` and `supybot.upkeepInterval` are not overridden in prod
  `bot.conf`. If `flush` is false, the automatic cleanup does not happen and the
  manual sweep becomes mandatory.
- Copy `bot.conf` off the host and record KEY-A through KEY-D in a password
  manager. Limnoria's rolling `bot.conf.backup.<timestamp>` files roll forward
  and will all be post-deletion within a couple of flush cycles.
- Record the deployed image tag: `docker inspect vibebot --format '{{.Config.Image}}'`.

**Steps:**

1. **Populate the env file.** `EnvironmentFile=-` (`vibebot.service:12`) and
   `--env-file` (`:19`) read the same file with **different parsers**, and only
   `--env-file` reaches Python. `docker run --env-file` is the strict one: no
   quotes (they become part of the value), no trailing comments, no spaces around
   `=` (a hard error), LF endings. A malformed file makes `docker run` exit, and
   `Restart=always` with no `HEALTHCHECK` turns that into a silent crashloop —
   this step, not the code change, is the highest outage risk. Avoid a stray
   `IMAGE=` line: `Environment=IMAGE=` precedes `EnvironmentFile=` in the unit.
   Set `GEMINI_API_KEY` (KEY-A) and `XAI_API_KEY` (**KEY-B**, confirmed
   canonical). Give `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` real values or delete
   them; a wrong-length placeholder is worse than an absent variable.
   Restart, then verify **from inside the container** by comparing a hash, never
   by printing key material:
   `docker exec vibebot python3 -c "import hashlib,os;k=os.environ.get('XAI_API_KEY','');print(len(k), hashlib.sha256(k.encode()).hexdigest()[:12])"`
2. **Deploy the code.** Watch for the startup line reporting how many variables
   redaction covers.
3. **Verify within the first hour**, before the upkeep flush rewrites `bot.conf`:
   `@ask` on a grok channel, `@ask` on the global Gemini path, `@draw`, and one
   grounded-search request.
4. **Sweep `bot.conf` by prefix, with the bot stopped** — every
   `supybot.plugins.LLM.*ApiKey*` line, including the stale `askApiKey`/
   `drawApiKey` entries. Expect the flush to have removed most already.
5. **Revoke** KEY-C and KEY-D, plus whatever the stale `askApiKey`/`drawApiKey`
   entries held.

**Rollback.** Before step 2, revert the env file. After step 2 the registry
settings no longer exist, so restoring a `bot.conf` entry does nothing. A revert
push is 10-20 minutes and can be blocked by one flaky test, since `docker.yml`
gates on CI across a three-version matrix. The fast path is an image pin — every
build is tagged `type=sha` (`docker.yml:41-46`) — and it must be written into
`operations.md` before step 2:

```
systemctl --user stop vibebot-updater.timer
mkdir -p ~/.config/systemd/user/vibebot.service.d
printf '[Service]\nEnvironment=IMAGE=ghcr.io/rdrake/vibebot:sha-<PREV>\n' \
  > ~/.config/systemd/user/vibebot.service.d/override.conf
systemctl --user daemon-reload && systemctl --user restart vibebot
```

Stopping the updater timer first is required: `vibebot-updater.service` hardcodes
`:latest` and would otherwise bounce the pinned container every 15 minutes.

## New exposure accepted

Environment variables are visible where `bot.conf` was not: `docker inspect`
returns them under `.Config.Env` (readable by anyone in the `docker` group, and
routinely pasted into tickets), `/proc/<pid>/environ` on the host, and child
processes inherit them — `plugin.py:1428` shells out for a git SHA. `journalctl`
is unaffected, since `ExecStart` passes a file path rather than `-e` pairs. This
is accepted: it trades a broad-but-shallow exposure for the elimination of key
drift, and the redaction filter closes the path that reaches users.

## Out of scope

- **Named model sets** (`fun` / `professional`). Once keys are provider-scoped,
  making a channel "fun" is `assistantModel.#chan` plus `verseModel.#chan`. If
  revisited: **`Profile` is already taken.** `profile.py` defines the *mode* axis
  (`chat`/`code`/`draw`/`verse`/`remind_action`); a channel-personality axis is
  orthogonal and needs a different name (`modelSet`, `tier`).
- **`assistantModel`'s `-latest` alias default.** `gemini/gemini-flash-latest`
  silently re-points across model versions. A real problem, unrelated to keys.
- **Credential-family aliases.** LiteLLM's provider label is a routing backend,
  not a credential family: `text-completion-openai` and `anthropic_text` are
  distinct labels that want the same key as their base provider. Nothing in this
  deployment uses them, and under the delegate-the-rest rule they fall through to
  LiteLLM's own resolution rather than failing. Add aliases if that changes.

# MkDocs User & Operator Guide — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-page help site with a full MkDocs Material documentation site covering user commands and operator configuration, deployed on GitHub Pages.

**Architecture:** MkDocs Material site with 10 markdown pages (index + 4 user guide + 4 operator guide + 1 reference). Built by GitHub Actions on push to main, deployed to `rdrake.github.io/vibebot/`. MkDocs is added as a dev dependency. The existing `build_help_page.py` script and `_site/` output are removed. MkDocs source lives in `docs/guide/` to avoid conflicts with existing `docs/plans/` and `docs/reviews/` directories.

**Note:** The reference page is hand-written markdown, not generated from `COMMAND_REGISTRY`. This means it can drift — when adding or changing commands, update `docs/guide/reference/commands.md` to match.

**Tech Stack:** MkDocs Material, GitHub Pages, uv (dev dependency group)

**Design doc:** `docs/plans/2026-04-07-mkdocs-guide-design.md`

---

### Task 1: Scaffold MkDocs and add dependencies

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/guide/index.md` (placeholder)
- Modify: `pyproject.toml` (dev dependency group)
- Modify: `Makefile` (add `docs` and `docs-serve` targets)

**Step 1: Add mkdocs-material to dev dependencies**

In `pyproject.toml`, add `mkdocs-material` to the `[dependency-groups] dev` list:

```toml
[dependency-groups]
dev = [
    "mkdocs-material>=9.6",
    "prek>=0.3.1",
    "pytest>=9.0.2",
    "pytest-cov>=7.0.0",
    "pytest-mock>=3.14.0",
    "ruff>=0.14.9",
    "ty>=0.0.1a34",
]
```

**Step 2: Install the new dependency**

Run: `uv sync`
Expected: mkdocs-material and its dependencies installed successfully.

**Step 3: Create `mkdocs.yml`**

```yaml
site_name: VibeBot v8
site_url: https://rdrake.github.io/vibebot/
repo_url: https://github.com/rdrake/vibebot
docs_dir: docs/guide
strict: true

theme:
  name: material

nav:
  - Home: index.md
  - User Guide:
    - Getting Started: user/getting-started.md
    - AI Commands: user/ai-commands.md
    - Memory & Instructions: user/memory.md
    - Reminders & Usage: user/reminders-usage.md
  - Operator Guide:
    - Installation & Deployment: operator/installation.md
    - Configuration: operator/configuration.md
    - Rate Limiting & Security: operator/rate-limiting-security.md
    - Tuning & Monitoring: operator/tuning-monitoring.md
  - Reference:
    - Command Reference: reference/commands.md
```

`docs_dir: docs/guide` is critical — it keeps MkDocs from picking up `docs/plans/`, `docs/reviews/`, and `docs/operations.md` which would cause strict-mode build failures.

**Step 4: Create placeholder `docs/guide/index.md`**

```markdown
# VibeBot v8

An IRC bot with AI capabilities, built on [Limnoria](https://github.com/ProgVal/Limnoria) and powered by [LiteLLM](https://github.com/BerriAI/litellm).

Placeholder — content coming in later tasks.
```

**Step 5: Add Makefile targets**

Add to `Makefile` (after the `help` target's echo lines, before the Docker section):

```makefile
docs:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve
```

Also add `docs docs-serve` to the `.PHONY` line and add help entries:

```
@echo "  docs            - Build MkDocs site"
@echo "  docs-serve      - Serve docs locally with live reload"
```

**Step 6: Verify the site builds**

Run: `uv run mkdocs build`
Expected: Build succeeds. Strict mode will warn about missing nav pages — that's expected since they'll be created in subsequent tasks. Use `mkdocs build` (without `--strict`) for this initial check only.

Note: If `mkdocs-material` fails to install on Python 3.14 (missing wheels for dependencies), fall back to running `uv run --python 3.13 mkdocs build` or pin a compatible Python in the docs workflow.

**Step 7: Commit**

```bash
git add mkdocs.yml docs/guide/index.md pyproject.toml uv.lock Makefile
git commit -m "docs: scaffold MkDocs Material site"
```

---

### Task 2: Write User Guide pages

**Files:**
- Create: `docs/guide/user/getting-started.md`
- Create: `docs/guide/user/ai-commands.md`
- Create: `docs/guide/user/memory.md`
- Create: `docs/guide/user/reminders-usage.md`

**Important context:**
- The actual command prefix is `@` (not `%` as in code comments — see memory `feedback_command_prefix.md`).
- Link to Limnoria docs for capabilities: `https://docs.limnoria.net/use/capabilities.html`
- All examples should use `@` prefix.

**Step 1: Write `docs/guide/user/getting-started.md`**

Content should cover:
- What VibeBot does (one paragraph — AI-powered IRC bot, multi-provider)
- Command prefix: `@` (e.g., `@ask`, `@code`, `@draw`)
- NickServ: some commands (like `@draw`) require a registered account
- Permissions: your bot operator grants access via Limnoria capabilities — link to [Limnoria capabilities docs](https://docs.limnoria.net/use/capabilities.html)
- Rate limits: exist to prevent abuse, your tier depends on your account status
- Quick command overview table (name, one-line description) linking to relevant pages

**Step 2: Write `docs/guide/user/ai-commands.md`**

Content should cover three commands on one page, each as a section:

`@ask`:
- Basic usage and examples
- Follow-up questions (context carries over automatically)
- Vision: include image URLs and the bot analyzes them
- Mention `@instruct` for persistent instructions (details on Memory page)

`@code`:
- Basic usage and examples
- Iterating on code (context carries over)
- Output as HTTP link with syntax highlighting
- Fallback to IRC paging if HTTP unavailable

`@draw`:
- Basic usage and examples
- Requires NickServ account
- Safety filter: if prompt is blocked, bot automatically rewrites and retries

**Step 3: Write `docs/guide/user/memory.md`**

Content should cover:

Volatile context:
- Automatic — conversation history maintained per user per channel
- Expires after inactivity (default 5 minutes)
- `@forget` clears it manually

Non-volatile memory:
- Facts extracted automatically from conversations
- `@memories` — list your stored facts
- `@memories delete <id>` — remove a fact
- `@memories edit <id> <text>` — correct a fact
- `@memories clear` — delete all
- `@memories cleanup` — deduplicate

Custom instructions:
- `@instruct <text>` — set persistent instructions for `@ask`
- `@instruct clear` — remove
- `@instruct` — show current

**Step 4: Write `docs/guide/user/reminders-usage.md`**

Content should cover:

`@remind`:
- Natural language time parsing (examples: "in 30 minutes", "at 5pm", "tomorrow at 9am")
- `@remind list` — show active reminders
- `@remind delete <id>` — cancel one or more
- `@remind clear` — cancel all
- Delivered via PM

`@usage`:
- `@usage` in a channel — your stats + channel stats
- `@usage someone` — another user's stats
- `@usage #channel` — a channel's stats

**Step 5: Verify build**

Run: `uv run mkdocs build --strict`
Expected: Build succeeds with no warnings.

**Step 6: Commit**

```bash
git add docs/guide/user/
git commit -m "docs: add user guide pages"
```

---

### Task 3: Write Operator Guide pages

**Files:**
- Create: `docs/guide/operator/installation.md`
- Create: `docs/guide/operator/configuration.md`
- Create: `docs/guide/operator/rate-limiting-security.md`
- Create: `docs/guide/operator/tuning-monitoring.md`

**Important context:**
- Link to Limnoria docs instead of duplicating. Key URLs:
  - Getting started: `https://docs.limnoria.net/use/getting_started.html`
  - Configuration: `https://docs.limnoria.net/use/configuration.html`
  - Capabilities: `https://docs.limnoria.net/use/capabilities.html`
  - HTTP server: `https://docs.limnoria.net/use/httpserver.html`
- Link to LiteLLM providers: `https://docs.litellm.ai/docs/providers`
- Config values are from `plugins/llm/src/llm/config.py` — use exact registry names.
- Docker image: `ghcr.io/rdrake/vibebot`

**Step 1: Write `docs/guide/operator/installation.md`**

Content should cover:
- Prerequisites: Docker (recommended) or Python 3.12+ with uv
- Link to [Limnoria getting started](https://docs.limnoria.net/use/getting_started.html) for initial `bot.conf` creation
- Docker deployment: `docker pull ghcr.io/rdrake/vibebot:latest`, volume mounts
- systemd service: `make install-service`, `make install-timer` for auto-updates
- Manual/dev setup: `make install && make run`
- Verifying: `systemctl --user status vibebot`

**Step 2: Write `docs/guide/operator/configuration.md`**

Content should cover:
- How to set config: link to [Limnoria configuration docs](https://docs.limnoria.net/use/configuration.html)
- API keys: `askApiKey`, `codeApiKey`, `drawApiKey` (private, per-command)
  - Fallback keys: `memoryApiKey` falls back to `askApiKey`, `spontaneousApiKey` falls back to `askApiKey`
- Model selection: `askModel`, `codeModel`, `drawModel` — link to [LiteLLM providers](https://docs.litellm.ai/docs/providers)
- System prompts: `askSystemPrompt`, `codeSystemPrompt` (channel-overridable)
- Per-channel overrides: most settings support `config channel #chan plugins.LLM.settingName value`
- `helpUrl` config (will point to the new MkDocs site)

**Step 3: Write `docs/guide/operator/rate-limiting-security.md`**

Content should cover:

Rate limiting:
- `enforceRateLimits` (True = enforce, False = shadow/monitor mode)
- Tiers: owner/admin (exempt), trusted, registered, unregistered
- Per-command settings: `{cmd}RateLimitCount`, `{cmd}RateLimitWindow`, `{cmd}TrustedRateLimitCount`, etc.
- Default limits table (ask: 15/60s, code: 10/60s registered / 0 trusted / 2 unreg, draw: 2/300s)
- Granting `trusted` capability: link to [Limnoria capabilities](https://docs.limnoria.net/use/capabilities.html)

Security:
- Capabilities required: `llm.ask`, `llm.code`, `llm.draw`
- NickServ gating for `@draw`
- URL validation (blocks non-HTTP schemes, path traversal, private IPs)
- Output sanitization (IRC command injection prevention, `commandPrefixes`)
- Prompt injection defenses (system prompt structure)

**Step 4: Write `docs/guide/operator/tuning-monitoring.md`**

Content should cover:

Context tuning:
- `contextEnabled`, `contextMaxMessages`, `contextTimeoutMinutes`
- `contextTrackAllMessages` (privacy implications — sends messages to LLM providers)
- `channelContextMaxMessages`

Memory tuning:
- `memoryEnabled`, `memoryMaxPerUser`, `memoryCleanupInterval`
- `memoryExtractionModel`, `memoryCleanupModel`

Spontaneous participation:
- `spontaneousEnabled`, `spontaneousChance`, `spontaneousCooldown`
- `spontaneousModel`, `spontaneousSystemPrompt`
- Requires `contextTrackAllMessages`

HTTP output:
- `httpRoot`, `httpUrlBase` — built-in vs external server
- Link to [Limnoria HTTP server docs](https://docs.limnoria.net/use/httpserver.html)
- `fileCleanupAge`, `fileCleanupMax`

Monitoring:
- Log locations: `journalctl --user -u vibebot`, bot's `logs/messages.log`
- `logLevel` setting (DEBUG for verbose tracing)
- Database: `databasePath` (default `data/LLM.db`)
- Common issues: API key problems, context not working, HTTP output not saving

**Step 5: Verify build**

Run: `uv run mkdocs build --strict`
Expected: Build succeeds with no warnings.

**Step 6: Commit**

```bash
git add docs/guide/operator/
git commit -m "docs: add operator guide pages"
```

---

### Task 4: Write Reference page (fold in existing help page)

**Files:**
- Create: `docs/guide/reference/commands.md`

**Important context:**
- The existing help page content comes from `COMMAND_REGISTRY` in `plugins/llm/src/llm/plugin.py:84`.
- The reference page is hand-written markdown (not generated). Keep it simple and maintainable.
- Use the command data from the registry but write it as a clean markdown table + per-command details.

**Step 1: Write `docs/guide/reference/commands.md`**

Start with a quick-reference table:

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@ask` | `<question>` | Ask the AI a question |
| `@code` | `<request>` | Generate code |
| `@draw` | `<prompt>` | Generate an image |
| `@forget` | `[channel]` | Clear conversation context |
| `@memories` | `[del <id> \| edit <id> <text> \| clear \| cleanup]` | Manage stored facts |
| `@instruct` | `[<instruction> \| clear]` | Set persistent instructions |
| `@remind` | `[<text> \| list \| del <id> \| clear]` | Set and manage reminders |
| `@usage` | `[nick \| #channel]` | Show API usage stats |

Then a Features section (from existing help page):
- Volatile memory for follow-up questions
- Non-volatile memory across conversations
- Vision support with image URLs
- Syntax-highlighted code output
- Spontaneous channel participation
- Multi-provider AI via LiteLLM

**Step 2: Verify build**

Run: `uv run mkdocs build --strict`
Expected: Build succeeds, all nav entries resolve.

**Step 3: Commit**

```bash
git add docs/guide/reference/
git commit -m "docs: add command reference page"
```

---

### Task 5: Update GitHub Pages workflow and clean up old help page

**Files:**
- Modify: `.github/workflows/pages.yml`
- Delete: `scripts/build_help_page.py`
- Delete: `_site/index.html`
- Modify: `plugins/llm/tests/test_plugin.py` (remove `TestBuildHelpPage` class)
- Modify: `.gitignore` (add `site/`, remove `_site/`)

**Step 1: Update `.github/workflows/pages.yml`**

Replace the workflow to use MkDocs instead of the build script:

```yaml
name: Deploy Docs

on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v6

      - name: Install uv
        uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true
          python-version: "3.14"

      - name: Install dependencies
        run: uv sync

      - name: Build docs
        run: uv run mkdocs build

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

Key changes from old workflow:
- `uv run mkdocs build` instead of `uv run python scripts/build_help_page.py`
- Upload `site` directory instead of `_site`
- `strict: true` is set in `mkdocs.yml` so no need for `--strict` flag

Note: If mkdocs-material dependencies don't have Python 3.14 wheels, change `python-version` to `"3.13"` — the docs build doesn't need to match the bot's runtime version.

**Step 2: Delete old files and remove stale test**

```bash
rm scripts/build_help_page.py
rm _site/index.html
rmdir _site
```

Also remove the `TestBuildHelpPage` class from `plugins/llm/tests/test_plugin.py` (around line 2316). This test imports and runs the deleted build script via subprocess — leaving it will break `make preflight`.

**Step 3: Add `site/` to `.gitignore`**

Check if `.gitignore` exists. If so, add `site/` to it. If not, create it with:

```
site/
```

Also remove `_site/` from `.gitignore` if present.

**Step 4: Update `docs/guide/index.md` with real content**

Replace the placeholder from Task 1 with a proper landing page:
- One paragraph describing VibeBot
- Links to User Guide and Operator Guide sections
- Links to GitHub repo and Limnoria

**Step 5: Verify build**

Run: `uv run mkdocs build --strict`
Expected: Build succeeds, `site/` directory created with all pages.

**Step 6: Verify locally**

Run: `uv run mkdocs serve`
Expected: Site serves at `http://127.0.0.1:8000/vibebot/` with working navigation.

**Step 7: Commit**

```bash
git add .github/workflows/pages.yml .gitignore docs/guide/index.md plugins/llm/tests/test_plugin.py
git rm scripts/build_help_page.py _site/index.html
git commit -m "docs: switch GitHub Pages to MkDocs, remove old help page"
```

---

### Task 6: Final verification and cleanup

**Files:**
- Modify: `CLAUDE.md` (update repository structure, remove `_site/` and `scripts/` references, add `mkdocs.yml` and `docs/guide/`)

**Step 1: Run preflight**

Run: `make preflight`
Expected: All checks pass (lint, format, typecheck, tests).

**Step 2: Check for stale references**

Search the codebase for references to `build_help_page`, `_site`, or the old help page workflow and update/remove them. Key files to check:
- `CLAUDE.md` — update the repository structure table to show `mkdocs.yml` and `docs/guide/` instead of `_site/` and `scripts/`. Add `make docs` and `make docs-serve` to the development commands section.
- Any remaining references in test files (the `TestBuildHelpPage` class should already be removed in Task 5).

**Step 3: Full docs build**

Run: `make docs`
Expected: Build succeeds with zero warnings.

**Step 4: Local preview**

Run: `make docs-serve`
Expected: All pages render correctly, navigation works, links resolve.

**Step 5: Commit any fixups**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for MkDocs migration"
```

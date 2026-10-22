# MkDocs User & Operator Guide — Design

## Goal

Replace the single-page help site with a full MkDocs Material documentation site covering both end-user commands and operator deployment/configuration. Deploy on GitHub Pages at the same URL (`rdrake.github.io/vibebot/`).

## Principles

- **Don't duplicate Limnoria docs.** Link to `docs.limnoria.net` for bot configuration basics, capabilities, plugin management, and HTTP server setup.
- **Two audiences, one site.** Users and operators get separate nav sections so neither has to wade through the other's content.
- **Fold the existing help page** into a Reference section (command quick-reference) so nothing is lost.
- **Minimal theme.** MkDocs Material defaults — no custom branding.
- **Separate docs directory.** MkDocs source lives in `docs/guide/` (via `docs_dir` config) to avoid conflicts with existing `docs/plans/` and `docs/reviews/`.

## Site Structure

```
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

## Page Summaries

### Home (`index.md`)

One-paragraph description of VibeBot. Links to User Guide and Operator Guide. Links to GitHub repo and Limnoria docs.

### User Guide

#### Getting Started

What VibeBot is from a user's perspective. Command prefix (`@`). NickServ registration requirement for some commands. Link to [Limnoria capabilities](https://docs.limnoria.net/use/capabilities.html) for how permissions work.

#### AI Commands (`@ask`, `@code`, `@draw`)

All three generation commands on one page — they share patterns (context, follow-ups). Covers: basic usage, multi-turn conversations, image URLs for vision, code output links, image generation with NickServ requirement and safety filter retries. Examples for each.

#### Memory & Instructions

Two memory systems explained together: volatile context (automatic, clears on timeout or `@forget`) and non-volatile facts (extracted automatically, managed with `@memories`). Persistent custom instructions with `@instruct`. Subcommands: list, delete, edit, clear, cleanup.

#### Reminders & Usage

`@remind`: natural language time parsing examples, managing reminders (list, delete, clear), PM delivery. `@usage`: checking personal and channel stats, what the numbers mean.

### Operator Guide

#### Installation & Deployment

Prerequisites (Docker, or Python 3.12+ with uv). Docker deployment with systemd. Auto-update timer. Link to [Limnoria getting started](https://docs.limnoria.net/use/getting_started.html) for initial bot.conf setup.

#### Configuration

API keys (how to set, why they're private). Model selection per command. Link to [LiteLLM provider docs](https://docs.litellm.ai/docs/providers) for models. Link to [Limnoria configuration docs](https://docs.limnoria.net/use/configuration.html) for `bot.conf` and the `config` command. Per-channel overrides. System prompts.

#### Rate Limiting & Security

Three-tier system (unregistered, registered, trusted, owner/admin). Configuring limits per command. Shadow mode for testing. Capabilities and access control (link to [Limnoria capabilities](https://docs.limnoria.net/use/capabilities.html)). URL validation, output sanitization, prompt injection defenses.

#### Tuning & Monitoring

Volatile context settings (max messages, timeout, channel tracking). Non-volatile memory limits and cleanup. Spontaneous participation. HTTP output setup (link to [Limnoria HTTP server docs](https://docs.limnoria.net/use/httpserver.html)). Log locations, common issues, database.

### Reference

#### Command Reference

Quick-reference table of all commands with syntax, description, and required capability. Replaces the existing GitHub Pages help page content. Kept in sync with the `COMMAND_REGISTRY`.

## Deployment

- Replace the current `pages.yml` workflow to build MkDocs instead of running `build_help_page.py`.
- Keep `scripts/build_help_page.py` for now (it feeds the command registry) — or generate the reference page from the same data during MkDocs build.
- Site deploys to the same `rdrake.github.io/vibebot/` URL.

## Limnoria Doc Links

Instead of documenting these topics, link to:

| Topic | URL |
|-------|-----|
| Getting started | https://docs.limnoria.net/use/getting_started.html |
| Bot configuration | https://docs.limnoria.net/use/configuration.html |
| Capabilities | https://docs.limnoria.net/use/capabilities.html |
| Plugin management | https://docs.limnoria.net/use/plugins/index.html |
| HTTP server | https://docs.limnoria.net/use/httpserver.html |

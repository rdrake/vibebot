# GitHub Pages Help Page Design

## Goal

Host the generated help page on GitHub Pages at `https://rdrake.github.io/vibebot/` so it stays in sync with the command registry and is publicly accessible. Remove the non-functional dynamic help page serving from the Limnoria HTTP callback.

## Architecture

A GitHub Actions workflow deploys the help page on every push to main. A small build script imports `HELP_HTML_TEMPLATE` from the plugin and writes it to `_site/index.html`. GitHub Pages serves it at the default URL.

The bot's `getPluginHelp()` links to a configurable `helpUrl` config option, defaulting to the GitHub Pages URL.

## Changes

### New files

- `.github/workflows/pages.yml` — deploys help page to GitHub Pages on push to main
- `scripts/build_help_page.py` — imports `HELP_HTML_TEMPLATE`, writes `_site/index.html`

### Modified files

- `plugins/llm/src/llm/config.py` — add `helpUrl` config option defaulting to `https://rdrake.github.io/vibebot/`
- `plugins/llm/src/llm/plugin.py`:
  - Remove `_serve_help_page()` method from `LLMHTTPCallback`
  - Remove `_get_help_url()` method from `LLM` class
  - Remove `_HELP_HTML_HEAD`, `_HELP_HTML_FOOT`, `_build_help_html()`, `HELP_HTML_TEMPLATE`, `_HELP_HTML_BYTES` (no longer needed at runtime — the build script imports the registry directly)
  - Update `getPluginHelp()` to use the new `helpUrl` config value
  - Keep `COMMAND_REGISTRY` and `CommandInfo` (still used by `getPluginHelp()` and the build script)
- `plugins/llm/tests/test_plugin.py` — update tests: remove HTML template tests, update help page serving tests
- `plugins/llm/tests/test_html_output.py` — remove `TestHelpPageStructure` (HTML tested in build script or CI instead)

### No changes

- Nginx config — `/llm/` keeps serving images/code as before
- `_CATEGORY_LABELS` and HTML generation logic move to the build script

## Workflow: pages.yml

```yaml
name: Deploy Help Page

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
      - uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true
          python-version: "3.14"
      - run: uv sync
      - run: uv run python scripts/build_help_page.py
      - uses: actions/upload-pages-artifact@v3
        with:
          path: _site
      - id: deployment
        uses: actions/deploy-pages@v4
```

## Build script: scripts/build_help_page.py

Imports `COMMAND_REGISTRY` from the plugin module, generates the HTML (reusing the same `_build_help_html()` logic), and writes `_site/index.html`.

## Config option

```python
conf.registerGlobalValue(
    LLM,
    "helpUrl",
    registry.String(
        "https://rdrake.github.io/vibebot/",
        _("""URL to the help documentation page."""),
    ),
)
```

## Decisions

- **Single page only** — no docs site framework, just the generated HTML
- **Deploy on every push to main** — simple, always current
- **Remove dynamic serving** — `_serve_help_page()` was not reachable in production (nginx serves static files)
- **Configurable help URL** — defaults to GitHub Pages URL, changeable for forks/self-hosted

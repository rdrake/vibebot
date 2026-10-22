# GitHub Pages Help Page Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy the generated help page to GitHub Pages and remove the non-functional dynamic help serving from the plugin.

**Architecture:** A build script imports `COMMAND_REGISTRY` from the plugin and generates the HTML help page (reusing the same generation logic currently in plugin.py). A GitHub Actions workflow runs the script and deploys to GitHub Pages on every push to main. The plugin's `getPluginHelp()` uses a new `helpUrl` config option instead of computing it from the HTTP server config.

**Tech Stack:** Python, GitHub Actions, GitHub Pages

**Before starting:** Enable GitHub Pages on the repo (source: GitHub Actions) via: `gh api repos/rdrake/vibebot/pages -X POST -f build_type=workflow`

---

### Task 1: Create the build script

**Files:**
- Create: `scripts/build_help_page.py`
- Test: Run the script directly

**Step 1: Create the build script**

Create `scripts/build_help_page.py`:

```python
#!/usr/bin/env python3
"""Build the HTML help page from the command registry.

Used by the GitHub Pages deployment workflow to generate _site/index.html.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Add plugin source to path so we can import the registry
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "llm" / "src"))

from llm.plugin import COMMAND_REGISTRY  # noqa: E402

_CATEGORY_LABELS = {"generation": "Generation", "memory": "Memory", "utility": "Utility"}

_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Bot Commands</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    padding: 20px;
    background: #272822;
    color: #f8f8f2;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    line-height: 1.6;
    max-width: 800px;
    margin: 0 auto;
}
h1 { color: #f8f8f2; margin-bottom: 0.5em; }
h2 { color: #a6e22e; margin-top: 1.5em; border-bottom: 1px solid #49483e; padding-bottom: 0.3em; }
h3 { color: #66d9ef; margin-top: 1.2em; }
code {
    font-family: 'SF Mono', 'Fira Code', Consolas, 'Liberation Mono', monospace;
    font-size: 14px;
    background: #1e1e1e;
    padding: 2px 6px;
    border-radius: 3px;
}
pre {
    background: #1e1e1e;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 1em 0;
}
pre code { padding: 0; background: none; }
.command { color: #e6db74; font-weight: bold; }
.param { color: #fd971f; }
.example { color: #75715e; font-style: italic; }
ul { margin: 0.5em 0; padding-left: 1.5em; }
li { margin: 0.3em 0; }
a { color: #66d9ef; }
.note {
    background: #3e3d32;
    border-left: 3px solid #a6e22e;
    padding: 10px 15px;
    margin: 1em 0;
    border-radius: 0 6px 6px 0;
}
@media (max-width: 600px) {
    body { padding: 15px; }
    pre { padding: 12px; font-size: 13px; }
}
</style>
</head>
<body>
<h1>LLM Bot Commands</h1>
<p>AI-powered IRC bot commands using LiteLLM.</p>
"""

_HTML_FOOT = """
<h2>Features</h2>
<ul>
<li><strong>Volatile Memory</strong> &ndash; Recent exchanges for natural follow-up questions (cleared by <code>%forget</code>, expires after timeout)</li>
<li><strong>Non-volatile Memory</strong> &ndash; Facts the bot remembers about you across conversations (managed by <code>%memories</code>)</li>
<li><strong>Vision Support</strong> &ndash; Include image URLs in <code>%ask</code> for image analysis</li>
<li><strong>Syntax Highlighting</strong> &ndash; Generated code is displayed with full highlighting</li>
<li><strong>Spontaneous Participation</strong> &ndash; The bot may occasionally join channel conversations (when enabled)</li>
<li><strong>Multi-Provider</strong> &ndash; Supports various AI providers via LiteLLM</li>
</ul>

<h2>Configuration</h2>
<div class="note">
Configuration is managed by the bot operator via Limnoria's config system.
Commands require the appropriate capability (e.g., <code>llm.ask</code>).
</div>

<p>Key settings include:</p>
<ul>
<li><strong>Model selection</strong> &ndash; Different models for ask/code/draw commands</li>
<li><strong>System prompts</strong> &ndash; Customize bot personality per command</li>
<li><strong>Context settings</strong> &ndash; Configure volatile memory limits</li>
</ul>

</body>
</html>"""


def _build_commands_html() -> str:
    """Build the command sections from the registry."""
    sections: list[str] = []
    for category in ("generation", "memory", "utility"):
        cmds = [c for c in COMMAND_REGISTRY if c.category == category]
        if not cmds:
            continue
        sections.append(f"<h2>{_CATEGORY_LABELS[category]}</h2>")
        for cmd in cmds:
            escaped_args = html.escape(cmd.args)
            sections.append(
                f'<h3><code class="command">%{cmd.name}</code> '
                f'<span class="param">{escaped_args}</span></h3>'
            )
            sections.append(f"<p>{html.escape(cmd.description)}</p>")
            example_lines = "\n".join(
                f'<span class="example">{html.escape(ex)}</span>' for ex in cmd.examples
            )
            sections.append(f"<pre><code>{example_lines}</code></pre>")
    return "\n".join(sections)


def main() -> None:
    """Generate _site/index.html from the command registry."""
    out_dir = Path("_site")
    out_dir.mkdir(exist_ok=True)

    page = _HTML_HEAD + _build_commands_html() + _HTML_FOOT
    out_file = out_dir / "index.html"
    out_file.write_text(page, encoding="utf-8")
    print(f"Wrote {out_file} ({len(page)} bytes)")


if __name__ == "__main__":
    main()
```

**Step 2: Run the script to verify it works**

```bash
uv run python scripts/build_help_page.py
cat _site/index.html | head -5
```

Expected: file is created, starts with `<!DOCTYPE html>`.

**Step 3: Add `_site/` to `.gitignore`**

Append `_site/` to `.gitignore` so the build output isn't committed.

**Step 4: Commit**

```bash
git add scripts/build_help_page.py .gitignore
git commit -m "feat: add build script for GitHub Pages help page"
```

---

### Task 2: Add GitHub Pages workflow

**Files:**
- Create: `.github/workflows/pages.yml`

**Step 1: Enable GitHub Pages**

```bash
gh api repos/rdrake/vibebot/pages -X POST -f build_type=workflow
```

If it fails because Pages is already enabled, that's fine.

**Step 2: Create the workflow**

Create `.github/workflows/pages.yml`:

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

      - name: Install uv
        uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true
          python-version: "3.14"

      - name: Install dependencies
        run: uv sync

      - name: Build help page
        run: uv run python scripts/build_help_page.py

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: _site

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

**Step 3: Commit**

```bash
git add .github/workflows/pages.yml
git commit -m "ci: add GitHub Pages deployment workflow"
```

---

### Task 3: Add `helpUrl` config option

**Files:**
- Modify: `plugins/llm/src/llm/config.py` (lines 360-384, HTTP Server Settings section)
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Write the failing test**

Add to `test_plugin.py` replacing the existing `test_get_plugin_help_includes_url` test (line 512):

```python
def test_get_plugin_help_uses_help_url_config(self, mocker: MockerFixture) -> None:
    """GIVEN helpUrl configured WHEN getPluginHelp called THEN uses that URL."""
    from llm.plugin import LLM

    mocker.patch("llm.plugin.httpserver")
    mocker.patch("llm.plugin.schedule")
    mocker.patch("llm.plugin.LLMDatabase")
    plugin = LLM(mocker.MagicMock())
    mocker.patch.object(plugin, "registryValue", return_value="https://rdrake.github.io/vibebot/")

    result = plugin.getPluginHelp()

    assert "https://rdrake.github.io/vibebot/" in result
    assert "ask" in result
```

**Step 2: Run to verify it fails**

```bash
make test
```

**Step 3: Add `helpUrl` config option**

Add after the `httpUrlBase` config (line 384) in `config.py`:

```python
conf.registerGlobalValue(
    LLM,
    "helpUrl",
    registry.String(
        "https://rdrake.github.io/vibebot/",
        _("""URL to the help documentation page. Shown in plugin help output."""),
    ),
)
```

**Step 4: Update `getPluginHelp()` in plugin.py**

Replace lines 1142-1160 (both `_get_help_url` and `getPluginHelp`) with:

```python
def getPluginHelp(self) -> str:  # noqa: N802
    """Return plugin help with documentation URL."""
    url = self.registryValue("helpUrl")
    names = ", ".join(cmd.name for cmd in COMMAND_REGISTRY)
    return _("AI-powered commands using LiteLLM. Commands: %s. Full documentation: %s") % (
        names,
        url,
    )
```

This removes `_get_help_url()` entirely.

**Step 5: Run tests**

```bash
make test
```

**Step 6: Commit**

```bash
git commit -m "feat: add helpUrl config, replace dynamic URL computation"
```

---

### Task 4: Remove dynamic help page serving from plugin

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Modify: `plugins/llm/tests/test_plugin.py`
- Modify: `plugins/llm/tests/test_html_output.py`

**Step 1: Remove from plugin.py**

Remove these blocks:
- `import html as _html` (line 7)
- `_HELP_HTML_HEAD` string (lines 185-243)
- `_CATEGORY_LABELS` dict (line 245)
- `_build_help_html()` function (lines 248-267)
- `_HELP_HTML_FOOT` string (lines 270-295)
- `HELP_HTML_TEMPLATE = ...` assembly line (line 297)
- `_HELP_HTML_BYTES = ...` (line 300)
- `_serve_help_page()` method (lines 321-346)
- The root-path branch in `doGet()` that calls `_serve_help_page` (lines 353-356)

After removing the root-path branch, `doGet` should return 404 for empty path instead:

```python
def doGet(self, handler: httpserver.RequestHandler, path: str) -> None:  # noqa: N802
    """Serve static files from LLM web directory."""
    # Remove leading slash
    path = path.lstrip("/")

    # No index page — help docs are on GitHub Pages
    if path == "":
        handler.send_response(404)
        handler.end_headers()
        return

    # Security: prevent directory traversal (early check before path operations)
    ...
```

**Step 2: Remove tests that reference removed code**

In `test_plugin.py`:
- Remove `TestHTTPCallbackServeHelpPage` class (lines 266-344, 4 tests)
- Remove `test_get_help_url_delegates_to_service` (lines 480-494)
- Remove `test_get_help_url_with_localhost_fallback` (lines 496-510)
- Remove `test_get_plugin_help_includes_url` (lines 512-529) — replaced by new test in Task 3
- Remove `TestHTMLHelpGeneration` class (lines 2407-2423, 2 tests)
- Update `test_doget_serves_help_at_root` (line 70-74) to expect 404 instead of 200:

```python
def test_doget_returns_404_at_root(self, http_callback, mock_handler: MagicMock) -> None:
    """GIVEN empty path WHEN doGet called THEN returns 404."""
    http_callback.doGet(mock_handler, "")
    mock_handler.send_response.assert_called_with(404)
```

In `test_html_output.py`:
- Remove `TestHelpPageStructure` class entirely (lines 345-389, 5 tests)

**Step 3: Run preflight**

```bash
make preflight
```

**Step 4: Commit**

```bash
git commit -m "refactor: remove dynamic help page serving, now on GitHub Pages"
```

---

### Task 5: Add build script test to CI

**Files:**
- Modify: `plugins/llm/tests/test_plugin.py`

**Step 1: Write a test that verifies the build script produces valid output**

Add to `test_plugin.py`:

```python
class TestBuildHelpPage:
    """Tests for the help page build script."""

    def test_build_script_generates_valid_html(self, tmp_path, monkeypatch) -> None:
        """GIVEN build script WHEN run THEN generates valid HTML with all commands."""
        monkeypatch.chdir(tmp_path)
        import importlib
        import scripts.build_help_page as build_mod

        importlib.reload(build_mod)
        build_mod.main()

        out = (tmp_path / "_site" / "index.html").read_text()
        assert out.startswith("<!DOCTYPE html>")
        assert "</html>" in out

        from llm.plugin import COMMAND_REGISTRY

        for cmd in COMMAND_REGISTRY:
            assert f"%{cmd.name}" in out, f"%{cmd.name} missing from built help page"
        assert "Generation" in out
        assert "Memory" in out
        assert "Utility" in out
```

Note: this may need a `conftest.py` addition or `sys.path` adjustment. If importing `scripts.build_help_page` doesn't work due to path issues, restructure as:

```python
import subprocess

class TestBuildHelpPage:
    """Tests for the help page build script."""

    def test_build_script_generates_valid_html(self, tmp_path) -> None:
        """GIVEN build script WHEN run THEN generates valid HTML with all commands."""
        result = subprocess.run(
            ["uv", "run", "python", "scripts/build_help_page.py"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[3]),
            env={**dict(__import__("os").environ), "_SITE_DIR": str(tmp_path / "_site")},
        )
        assert result.returncode == 0

        out = (tmp_path / "_site" / "index.html").read_text()
        assert out.startswith("<!DOCTYPE html>")
        assert "</html>" in out

        from llm.plugin import COMMAND_REGISTRY

        for cmd in COMMAND_REGISTRY:
            assert f"%{cmd.name}" in out
```

If using the env-var approach, update `build_help_page.py` `main()` to respect `_SITE_DIR`:

```python
def main() -> None:
    out_dir = Path(os.environ.get("_SITE_DIR", "_site"))
    ...
```

Choose whichever approach works cleanly. The key requirement is: a test that runs the build script and verifies the output contains all commands.

**Step 2: Run tests**

```bash
make test
```

**Step 3: Commit**

```bash
git commit -m "test: add build script output verification"
```

---

### Task 6: Final verification and push

**Step 1: Run full preflight**

```bash
make preflight
```

Expected: all tests pass, coverage >= 80%.

**Step 2: Push**

```bash
git push
```

**Step 3: Verify GitHub Pages deployment**

```bash
gh run list --limit 5
```

Wait for the "Deploy Help Page" workflow to complete, then verify:

```bash
curl -sI https://rdrake.github.io/vibebot/
```

Expected: HTTP 200 with `content-type: text/html`.

**Step 4: Verify the page content**

```bash
curl -s https://rdrake.github.io/vibebot/ | grep -o '%[a-z]*' | sort -u
```

Expected: all 8 commands listed.

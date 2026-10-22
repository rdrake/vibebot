# Contributing to VibeBot v8

## Getting started

### Prerequisites

Install [uv](https://github.com/astral-sh/uv), the Python package manager this project uses:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

1. Clone the repository.
2. Install dependencies: `make install`
3. Install git hooks: `make install-hooks`

## Git hooks

This project uses [prek](https://github.com/j178/prek), a fast Rust-based pre-commit runner, to catch issues before they reach the repo. `make install-hooks` installs them.

The hooks check:

- **Secrets** (gitleaks): blocks API keys, passwords, and tokens
- **Lint** (ruff): catches code issues and auto-fixes where possible
- **Format** (ruff): keeps style consistent
- **Types** (ty): catches type errors
- **Hygiene**: merge-conflict markers, files over 500 KB, trailing whitespace, missing final newline

If a check fails, fix the issue and commit again. Run `make pre-commit` to run the pre-commit hooks on every file, including ones you have not touched.

The config also defines a pre-push hook that runs `make check-fast` (lint, format-check, typecheck); the test suite stays in CI so pushes stay quick. `make install-hooks` installs the pre-commit shim only — run `uv run prek install -t pre-push` to catch a broken push before CI does.

## Code quality

Run the full gate before calling any change done:

```bash
make preflight  # format, then lint + format-check + typecheck + test
```

Individual checks:

```bash
make lint         # ruff check
make format       # ruff format
make typecheck    # ty
make test         # pytest, skipping slow tests
make test-all     # full suite, including slow tests
```

### Coverage

Tests must keep at least **93%** branch coverage; `make test` enforces the floor. For a detailed report:

```bash
uv run pytest plugins/llm/tests/ plugins/nickinmiddle/tests/ --cov --cov-report=term-missing
```

### Docs

The published guide builds from `docs/guide/` with MkDocs (`make docs`), and CI does not lint prose.

Prose linting with [Vale](https://vale.sh/) is optional and local-only: `styles/` and `.vale.ini` are gitignored, so a fresh clone has no Vale setup. To use it, install Vale, write a `.vale.ini` pointing `StylesPath` at `styles`, run `vale sync`, then `vale docs/guide README.md`. Keep new content free of errors; the warning and suggestion levels carry house rules that fight this project's voice.

## Commit messages

Write [conventional commits](https://www.conventionalcommits.org/): `type(scope): summary`. git-cliff generates the changelog from them, grouping by type:

| Prefix | Changelog section |
|--------|-------------------|
| `feat` | Features |
| `fix` | Bug Fixes |
| `perf` | Performance |
| `refactor` | Refactor |
| `docs` | Documentation |
| `test` | Tests |
| `ci` | CI |
| `build` | Build |
| `deps` | Dependencies |
| `chore` | Chores |
| `revert` | Reverts |

## Branches and pull requests

The maintainer commits to `main` directly. Outside contributions arrive as pull requests from feature branches, and CI must pass before merge:

- `lint`: the prek hooks (ruff, ruff format, ty, gitleaks, hygiene)
- `check (3.14 / 3.15)`: the full test suite on the image's Python and the next one
- `secrets`: gitleaks scanning
- `docker`: builds the container image on pull requests; advisory, never a required check

Pushes to `main` that pass CI build and publish the Docker image, then restart production automatically. Treat any push touching runtime files as a deploy. Docs-only pushes (`docs/**`, `*.md`, `mkdocs.yml`) skip CI by design, so they neither rebuild the image nor bounce the bot; `docs/**` and `mkdocs.yml` publish through the Pages workflow instead.

## Releases

Push a semver tag to cut a release:

```bash
git tag v1.2.3
git push origin v1.2.3
```

The release workflow generates notes with git-cliff, creates the GitHub release, and opens a PR that refreshes `CHANGELOG.md` on `main`. `make changelog` regenerates `CHANGELOG.md` locally, overwriting the tracked file; it needs `git-cliff` on `PATH` (`brew install git-cliff`).

## Dependencies

- Python 3.14 or newer (the image runs 3.14; CI also tests 3.15)
- [uv](https://github.com/astral-sh/uv) manages the workspace and lockfile
- Dependabot proposes weekly updates; fold lockfile bumps into a single `uv lock` rather than merging per-package PRs

# Branch protection (optional)

VibeBot's pipeline does not require GitHub branch protection. The upstream repository (`rdrake/vibebot-v8`) is single-maintainer; direct pushes to `main` are normal, and the CI, Docker build, and auto-deploy pipeline is the safety net. This page is for forks and team deployments that want a hard PR gate.

## Recommended ruleset

In **Settings → Rules → Rulesets**, add a branch ruleset targeting `main`. **Settings → Branches** works too if you prefer a classic protection rule. Every setting below exists in both editors, except that rulesets express *Do not allow bypassing* as an empty bypass list.

| Setting | Value |
|---------|-------|
| Require a pull request before merging | Enabled |
| Required approvals | 0 (solo) or 1 or more (team) |
| Require status checks to pass before merging | Enabled |
| Require branches to be up to date before merging | Enabled |
| Do not allow bypassing the above settings | Enabled |

## Required status checks

Add these checks after the first CI run on a PR; they do not appear in the picker before then:

- `check (3.14)`
- `check (3.15)`
- `lint`
- `secrets`

The `pull_request` trigger in `ci.yml` is unfiltered, unlike the `push` trigger's `paths-ignore`. Keep it that way: a required check that never reports leaves a docs-only pull request permanently unmergeable.

The `docker` job in `ci.yml` is deliberately not required. That job is a PR-only smoke build that never pushes; the production image build lives in `docker.yml`, gated separately on CI success.

## Optional settings

| Setting | When to use |
|---------|-------------|
| Require conversation resolution | Team workflows where review comments must be addressed before merge |
| Require signed commits | Hardened deployments; every contributor needs a signing key |
| Require linear history | If you prefer rebase or squash merges over merge commits |

## Why the docker check is advisory

For PRs, CI runs four jobs: `lint`, `check` (a 3.14/3.15 matrix), `secrets`, and `docker`. The `docker` job builds the image but never pushes. The production build and push is a separate workflow, `docker.yml`, triggered after CI succeeds on `main`.

**Risk accepted:** a PR that breaks the Dockerfile can merge. CI passes, `docker.yml` then fails on `main`, and the previous image keeps serving production until a fix lands.

**Why accept it:** Docker builds are slow and the Dockerfile changes rarely. Requiring the check would add latency to every merge to guard against a rare, quickly recoverable failure.

## Verifying the rule

1. Push a commit directly to `main` from a non-bypass account; it should be rejected.
2. Open a PR with a deliberately failing test; the merge button should be unavailable.
3. Fix the test; the merge button should re-enable once checks pass.

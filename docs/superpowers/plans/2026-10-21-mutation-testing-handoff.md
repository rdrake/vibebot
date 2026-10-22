# Mutation testing: handoff (2026-10-21)

Pick up from here in any session. Goal (owner): find out whether the test suite actually proves the code, or only gives a false sense of security.

> **Owner rule: the tests prove the code, not the other way around.** When a mutant survives, take the intended behaviour from the docs, the user-visible contract, or the incident the code guards, never from what the code does today. If the original code is what's wrong, report a code bug; don't write a test that pins it.

## State at hand-off

- **Shipped to main (pushed):**
  - `61c019e` removes video/@animate
  - `b683ac3` removes @code
  - `48b2f5e` thins the tests: about 690 duplicate, trivial, mock-mirror and prose-pin tests dropped, leaving 3,246 tests at 93.1% coverage
  - `9ac61b4` switches CI to Python 3.14 and 3.15 and requires 3.14
  - `01b8c38` applies ruff's py314 format (PEP 758 `except A, B:`)
  - Branch protection's required checks are now `check (3.14)`, `check (3.15)`, `secrets` and `docker`.
- **Prod:** deploys from `01b8c38` once CI and Docker go green. `pending_tasks` was empty, so there were no leftover animate/code rows. Still to do on prod, with the bot stopped: delete the orphaned `bot.conf` keys (`animate*`, `code*`, `imageFallback*`, `imageApiBase`, `imageSteps`, `imageSize`) and `ANIMATE_API_KEY` from the env file. They are harmless if left.
- **Mutation sweep:** running detached (`nohup`) over all 34 source modules, smallest first.
  - Work dir `~/.cache/vibebot-mutation/` holds `wt0`–`wt9` (worktrees at `48b2f5e`, each with its own venv), `cov.json`, `mut/<module>.json` results and `sweep.log`. The log ends with `SWEEPDONE` when finished.
  - `pre_thinning/` holds runs of agenda, provider, identity, usertz, executor and ircquery against the *pre-thinning* suite. Comparing them with `mut/` shows what the thinning cost.
  - The audit TSVs (`audit_*.tsv`) list every deleted test with its category.

## Resume

```
pgrep -f mutation_sweep.sh || (cd ~/.cache/vibebot-mutation && nohup ~/workspace/afternet/vibebot-v8/scripts/mutation_sweep.sh > sweep.log 2>&1 &)
tail -3 ~/.cache/vibebot-mutation/sweep.log
```

The sweep skips modules that already have a result, so an interrupted module simply restarts. After a reboot, or if the worktrees are gone, run `scripts/mutation_sweep.sh --setup 48b2f5e` first. Setup takes about 2 minutes: it builds 10 venvs and runs one coverage pass.

## How the harness works (`scripts/mutate.py`)

- **Mutations.** It makes AST mutations: comparison swaps (`<`↔`<=`, `==`↔`!=`, `in`↔`not in`, `is`↔`is not`), `and`↔`or`, dropping `not`, `+`↔`-` and similar, `int n→n+1`, `True`↔`False`, forcing an `if` to True or False, and `return x → return None`. It skips docstrings, decorators and logging lines.
- **Uncovered lines.** A mutant on a line no test executes is recorded as `uncovered` without being run. Module-level code runs at import, before coverage starts, so some of these are false "uncovered". `profile.py` (7) is one: treat it as unmeasured, not as untested.
- **Two tiers.** Tier 1 runs the test files that mention the enclosing function or class. If the mutant survives that, tier 2 runs every test file that imports the module.
- **Exit codes.** Exit 1 or 2 means `killed`, and `error4` (the mutant broke an import) counts as killed too. A run past 300 s is recorded as `timeout`.
- **Output.** Each record has `{idx, line, desc, scope, src, status, tier, secs}`. `idx` indexes `enumerate_mutants(ast.parse(source))`. To reproduce a mutant, run `mutant_source(src, idx)` from `scripts/mutate.py` against the worktree source at `48b2f5e`.

## Next: the evaluation workflow (owner asked for a dynamic workflow)

Once `SWEEPDONE` appears:

1. **Gather survivors.** Collect the `survived` and `uncovered` records per module from `mut/`. Split large modules into chunks of 40 or fewer by `scope`.
2. **Triage** (one agent per chunk, read-only). Classify each survivor:
   - `equivalent`: no observable difference.
   - `cosmetic`: wording, logs, or performance only.
   - `real-gap`, with severity high, med or low: behaviour that matters, and the intended behaviour is stated *with its source*. The source is the docs (`docs/guide`), AGENTS.md, GLOSSARY.md, the feature's purpose, or memory notes. It is never "what the code does".
   - `code-bug`: the original code looks wrong by the spec.
3. **Verify** each high or med `real-gap` and each `code-bug` with two skeptics using different lenses:
   - Is the mutant really observable, or is it equivalent?
   - Is the claimed intended behaviour grounded in the spec, or is it derived from the code?

   A finding is kept unless refuted.
4. **Write tests** (one agent per module, in an isolated worktree off **current main**). Write the tests that kill the confirmed gaps, asserting the intended behaviour. Prove each one passes on the original code and fails on the mutant. Put them in the existing test file for that module. Commit locally in the worktree. **Never push**: a push to main is a prod deploy. `code-bug` findings are reported, not "fixed" by a test.
5. **Synthesize** a report: the per-module kill rate with uncovered counted separately, confirmed gaps by severity, code bugs, and what the thinning cost (compared with `pre_thinning/`).

Early readings, partial: `meme_render.py` 137 of 273 survived, `github.py` 37 of 94 survived with 31 uncovered, `verse/compaction.py` 33 of 72 survived. `agenda.py`, `identity.py` and `tracing.py` kill nearly everything.

## Leftovers from the LiteLLM handoff (not started)

The operator docs rewrite: README, AGENTS.md, `plugins/llm/README.md`, `.env.example` and `docs/guide/operator/configuration.md` still describe LiteLLM, Gemini and OpenAI. `rate-limiting-security.md` (around lines 101–102) describes an anti-injection preamble that the assistant path never had.

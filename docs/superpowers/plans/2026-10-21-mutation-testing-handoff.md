# Mutation testing: handoff (2026-10-21)

Pick up from here in any session. Goal (owner): find out whether the test suite actually proves the code, or only gives a false sense of security.

> **Owner authorization (2026-10-21):** "you can edit and push as you'd like". The owner expects this pass to uncover many bugs. Fix real code bugs, not only tests, and push from the main session in reviewed batches, each with `make test-all` green. Subagents still never push: a push is a prod deploy.
>
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

## Sweeps moved to the OTU cluster (2026-10-22 05:13)

The local sweep is stopped. The jobs run with `scripts/mutation_slurm.sh` on gpu05, scavenge partition, `--requeue`.
- **Access:** connect the WireGuard VPN "Work", then `ssh fs01`.
- **Work dir:** `~/vibebot-mut/` holds the source tarballs, `mutate.py`, `runs/<RUN>/mut/` and `logs/`.

Jobs:
- **Job 2062 (A):** finishes `plugin.py` at `48b2f5e`. It resumed from the local checkpoint with 2,600 done, leaving 1,083 to run on 88 workers.
- **Job 2063 (B):** re-sweeps every module on main at `7f8412e`, the code with the 213 new tests, to measure how far the kill rates moved.

A job that dies resumes when resubmitted with the same arguments.

- **Job A:** DONE at 05:31. `plugin.py` at `48b2f5e` killed 2,930, survived 753 and left 421 uncovered, for a 79.6% kill rate.
- **Part-A evaluation:** `wf_48ed1eba-06a`, args in `eval/args_plugin_a.json`. It covers the 448 survivors from the first 2,500 mutants. DONE: merged and pushed at `41cac69`. 95 confirmed (19 high), 80 tests, 3 fixes:
  - notify_when_online comma lists;
  - lapsed watches free their MONITOR slot;
  - irc_lookup comma targets.

  The suite now stands at 3,544 tests and 95.2% coverage.
- **Job B:** DONE at 06:42. Results are in `~/.cache/vibebot-mutation/afterB/`. The overall kill rate rose from 68.3% to 72.4% counting uncovered mutants, and from 77.6% to 81.0% over covered mutants only. Survivors fell from 2,090 to 1,752 and uncovered from 1,274 to 1,099. The biggest gains: meme_render +32, ircquery +13, compaction +12. plugin.py moved only +0.5 because part A's tests landed after `7f8412e`. The one regression: `taste_mine` line 104, the `name[:1]` capitalisation, flipped from killed to survived.
- **Part-B evaluation:** `wf_138fa122-568`, args in `eval/args_plugin_b.json`. It covers the other 726 records: 305 survived and 421 uncovered.
  - DONE and on main as `cec7843..71d3b73`: 83 confirmed (6 high), 55 tests, and 1 fix (@meme with no catalog now fails once and spends nothing).
  - The suite now stands at 3,605 tests and 95.6% coverage.
  - The report is `eval/report_plugin_b.md`, and the result is `eval/result_plugin_b.json`.
  - 176 low gaps are left unverified.

**Round 2 (2026-10-22): DONE and on main.**
- **Scope:** `provider`, `persistence`, `github` and `taste_mine`, from the `afterB` survivors.
- **Result:** 8 medium gaps got tests and no code bugs were found. Everything left is low severity.
- **Docs audit:** 90 defects fixed in 24 files.
- **Report:** `eval/report_round2.md`. It wrongly calls 49 gaps "unadjudicated": they are all low severity, and the skeptics skip low findings by design.

**Final sweep: job 2144 (`C-final-ad26bfa`), RUNNING.**
- **Where:** gpu05, 96 cores, 512 GB. Results go to `fs01:~/vibebot-mut/runs/C-final-ad26bfa/mut/`.
- **When it logs SWEEPDONE:**
  - Rsync the results to `~/.cache/vibebot-mutation/final/`.
  - Compare per-module kill rates against `mut/` (68.3% overall, before any new tests) and `afterB/` (72.4%).
  - Evaluate only a module that shows new med/high survivors, then close this doc out.
- **Missing from the sweep:** the 4 provider tests in `e8069bb` landed after `ad26bfa`.

**All planned evaluation is complete.** What's left is backlog:
- the `provider.py` Anthropic reply and cost tests;
- the `taste_mine` line-104 regression;
- the low gaps;
- `github` and `persistence`, the weakest modules.

Future replay-heavy work goes on gpu05, not the laptop, because Defender plus the local replays froze the GUI.

## Evaluation results: 33 modules DONE and on main (2026-10-22)

- **Report:** `~/.cache/vibebot-mutation/eval/report.md`. The full result JSON is in `eval/result.json`.
- **Findings:** 329 confirmed gaps (19 high, 305 med, 5 low).
- **On main** at `24b3b33`, pushed: 213 tests and 4 code fixes. The fixes are scheduled-task `@instruct` as fenced data, blocked-draw rewriter spend, storybook cap 0, and verse tools no longer reporting ok when nothing was written.
- **Open:**
  - `plugin.py` evaluation, whose sweep is still running.
  - Anthropic reply and cost tests in `provider.py`: thinning left 21 mutants uncovered.
  - Re-mutate `meme_render`, `compaction`, `github`, `usertz` and `persistence`.
  - 611 unverified low gaps.
- **Script fix:** the evaluation script now treats a finding that a skeptic skipped as unverified.

## Evaluation workflow (first run, 2026-10-22)

- **Launched** on the 33 finished modules, everything except `plugin.py`, whose sweep is still running.
  - Run ID: `wf_49fc9b59-7da`.
  - Script: `~/.claude/projects/-Users-rdrake-workspace-afternet-vibebot-v8/caabfa35-39ea-4a12-94e5-1a4d95165c67/workflows/scripts/mutation-evaluate-wf_109520a7-a7a.js`.
  - Input: `~/.cache/vibebot-mutation/eval/chunks.json`, 74 chunks of 2,135 records.
- **Resume in the same session:** call Workflow with `{scriptPath, resumeFromRunId: "wf_49fc9b59-7da", args: {chunks: <contents of chunks.json>}}`.
- **New session:** re-run the script with the same args. Completed fix worktrees under `.claude/worktrees/` keep their commits.
- **The report** lands in `~/.cache/vibebot-mutation/eval/report.md`.
- **When `plugin.py` finishes:** chunk `mut/llm__plugin.json` the same way, with survived and uncovered grouped by scope and at most 40 per chunk. Run the same script on just those chunks.

## The evaluation workflow, as designed

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
4. **Write tests** (one agent per module, in an isolated worktree off **current main**). Write the tests that kill the confirmed gaps, asserting the intended behaviour. Prove each one passes on the original code and fails on the mutant. Put them in the existing test file for that module. Commit locally in the worktree. Subagents never push. The main session reviews each worktree's commits, cherry-picks them, runs `make test-all` and pushes. A `code-bug` finding gets a test asserting the *intended* behaviour (red on the current code), plus a code fix that turns it green. It is never fixed by pinning what the code does now.
5. **Synthesize** a report: the per-module kill rate with uncovered counted separately, confirmed gaps by severity, code bugs, and what the thinning cost (compared with `pre_thinning/`).

Early readings, partial: `meme_render.py` 137 of 273 survived, `github.py` 37 of 94 survived with 31 uncovered, `verse/compaction.py` 33 of 72 survived. `agenda.py`, `identity.py` and `tracing.py` kill nearly everything.

## Leftovers from the LiteLLM handoff (not started)

The operator docs rewrite: README, AGENTS.md, `plugins/llm/README.md`, `.env.example` and `docs/guide/operator/configuration.md` still describe LiteLLM, Gemini and OpenAI. `rate-limiting-security.md` (around lines 101–102) describes an anti-injection preamble that the assistant path never had.

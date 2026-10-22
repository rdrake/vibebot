---
name: writing-tests
description: Use when writing, changing or reviewing tests in this repo, fixing a bug, or deciding what a surviving mutant means. Covers where expected values come from, what to assert, which fakes to use, and the scoped mutation check that proves a new test kills something.
---

# Writing tests that prove the code

The 2026-10 mutation pass found 515 behaviours that 93% line coverage never asserted, plus 8 code bugs. Finding them after the fact took thousands of agent runs. Each lesson below is cheap to apply while you write the test.

## 1. Take the expected value from the contract

The **contract** is what the behaviour is meant to be. In order of authority, it comes from:

- `docs/guide/`, `AGENTS.md` and `GLOSSARY.md`;
- the incident or user report the code guards against;
- what an IRC user or the model actually sees.

The value the code returns today is never the contract. Write the assertion from the contract first, then run the test.

When the code disagrees with the contract, you have found a code bug. Keep the test red, fix the code, and say so in the commit. Check the contract twice before you change behaviour. In the evaluation, claims that "the code is wrong" were refuted about 40% of the time, against about 5% for claims that "this is untested". A test written from a misread contract changes prod behaviour for no reason.

## 2. Assert the observable effect at the seam

A test earns its place when it pins one effect that someone downstream depends on. These are the effects the old suite most often left unasserted:

- **Wire output.** The exact line sent to IRC, and where it went. A reply addressed to the bot's own nick in a PM is a real incident class.
- **The tool-result envelope.** `ok`, or `error` with its message, exactly as the model receives it. The model narrated verse acts that never happened because a failure path returned `ok=True`.
- **Spend.** The tokens and cost that get booked, including on failure, refusal and rewrite paths. A booked `$0` is usually a bug.
- **Stored rows.** What is in SQLite after the call, especially before anything gets deleted. Compaction must write the summary before it drops the event.
- **The provider request.** What is sent to the provider: which guidance block rides with which tool, `max_tokens`, `tool_choice`, and history trimming.
- **Guards.** Delete the guard and some test must go red. Guards here are the SSRF/URL checks, account gates and caps; a cap of `0` means none.
- **Boundaries and lifetimes.** `<` against `<=`, earliest across both tables, and "exactly one step" or "exactly once".
- **Whole algorithms.** Text fitting, crops and warps: assert numeric results on a known input, not merely that the code runs.

A test pins one behaviour no other test pins. Name it as a sentence stating that behaviour, such as `test_a_host_with_any_disallowed_address_is_never_contacted`.

The thinning removed about 690 tests that pinned nothing. Their kinds, so you recognise them:

- A **duplicate** of a path already covered.
- A **trivial** check that a constant or registry key exists.
- An **over-granular** split of one branch into many tests.
- A **prose-pin** that freezes prompt or help wording.
- A **mock-mirror** that asserts a mock was called with the arguments the test itself passed.

## 3. Use real collaborators; fake only the edges

- **Storage:** real SQLite on `tmp_path`. For the plugin fixture's mocked DB, `conftest.back_reminders_with_sqlite` routes the reminder calls to a real store.
- **Time:** a fake clock passed in. Never sleep and hope. `test_agenda.py` shows the pattern.
- **HTTP:** a local `ThreadingHTTPServer` with one path per behaviour. `test_pinned_fetch.py` shows the pattern. Inject `resolve`/`allow` rather than patching sockets.
- **Boundaries you can't run:** mock only the provider API and the IRC connection. Never mock the function under test or its own module's helpers. A mocked-out subject once hid a real bug under six passing tests.

Fixture traps that have cost hours:

- The registry stub (`make_registry_side_effect`) returns `""` for any key it doesn't know. A feature gated on a missing or misspelled key reads as off, and a test of the gated path can pass without the code under test ever running. Pass the key as an override, and assert an effect that only the gated path produces.
- A bare `MagicMock` standing in for a DB row or a number fails far from the cause. Return real values.
- supybot's logger has no `%d`; use `%i`. `test_log_format_specifiers.py` enforces it.

## 4. Prove each new test kills something

This step is done when every function you wrote or changed tests for has had its scoped mutation check. Each survivor must then be either killed by a test or written off in your summary with a one-line reason: *equivalent* (no observable difference) or *cosmetic* (wording, logs or performance only).

```
python3 scripts/mutate.py plugins/llm/src/llm/<module>.py --wt . \
  --scope <function> [--scope <other>] --out "$TMPDIR/mut.json"
python3 -c "import json,sys;[print(r['status'],r['line'],r['desc'],r['src'][:80]) for r in json.load(open(sys.argv[1])) if r['status']!='killed']" "$TMPDIR/mut.json"
```

- The check runs in this checkout, one mutant at a time, about 20 s a mutant. It restores the module after each mutant. If a run is interrupted, run `git diff` on the module before anything else.
- Pass the innermost function or class you touched. More than about 30 mutants is a job for the cluster sweep, not the laptop.
- Test the change itself (did a mutant survive?), not the backlog. A low-severity survivor in code you didn't touch stays where it is.

## 5. Full sweeps

A whole-codebase sweep runs on the OTU cluster with `scripts/mutation_slurm.sh`. The laptop runs `scripts/mutation_sweep.sh`, but its test churn freezes the GUI under Defender.

`docs/superpowers/plans/2026-10-21-mutation-testing-handoff.md` covers the triage → skeptic → test workflow and the last results. Treat an "uncovered" label from a sweep with suspicion: coverage line-mapping mislabelled about 30 mutants that a test in fact killed. Re-run the mutant before writing a test for it.

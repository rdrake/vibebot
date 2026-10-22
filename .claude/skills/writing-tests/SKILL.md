---
name: writing-tests
description: Use when writing, changing or reviewing tests in this repo, fixing a bug, or deciding what a surviving mutant means. Covers where expected values come from, what to assert, which fakes to use, and the scoped mutation check that proves a new test kills something.
---

# Writing tests that prove the code

The 2026-10 mutation pass found 515 behaviours that 93% line coverage never asserted, plus 8 code bugs. Finding them after the fact took days of sweeps and hundreds of agent runs. Each lesson below is cheap to apply while you write the test.

## 1. Take the expected value from the contract

The **contract** is what the behaviour is meant to be. In order of authority, it comes from:

- `docs/guide/`, `AGENTS.md` and `GLOSSARY.md`;
- the incident or user report the code guards against;
- what an IRC user or the model actually sees.

The value the code returns today is never the contract. Write the assertion from the contract first, then run the test.

When the code disagrees with the contract, you have found a code bug. Keep the test red, fix the code, and say so in the commit. Check the contract twice before you change behaviour. In the evaluation (`eval/report.md` in the results archive the handoff doc names), claims that "the code is wrong" were refuted about 40% of the time, against about 5% for claims that "this is untested". A test written from a misread contract changes prod behaviour for no reason.

## 2. Assert the observable effect at the seam

A test earns its place when it pins one effect that someone downstream depends on. These are the effects the old suite most often left unasserted:

- **Wire output.** The exact line sent to IRC, and where it went. A reply addressed to the bot's own nick in a PM is a real incident class.
- **The tool-result envelope.** `ok`, or `error` with its message, exactly as the model receives it. The model narrated verse acts that never happened because a failure path returned `ok=True`.
- **Spend.** The tokens and cost that get booked, including on failure, refusal and rewrite paths. A booked `$0` is usually a bug.
- **Stored rows.** What is in SQLite after the call, especially before anything gets deleted. Compaction must write the summary before it drops the event.
- **The provider request.** What is sent to the provider: which guidance block rides with which tool, `max_tokens`, `tool_choice`, and history trimming.
- **Guards.** Delete the guard and some test must go red. Guards here are the SSRF/URL checks, account gates and caps. Test each setting's `0` against its own contract, because `0` means different things for different settings: the storybook image cap at `0` means none, while a rate-limit count at `0` means unlimited.
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
- **Boundaries you can't run:** mock only the provider API and the IRC connection. The behaviour under test runs for real, including any helpers it relies on. A mocked-out subject once hid a real bug under six passing tests.

Fixture traps that have cost hours:

- The registry stub (`make_registry_side_effect`) returns `""` for any key it doesn't know. A feature gated on a missing or misspelled key reads as off, and a test of the gated path can pass without the code under test ever running. Pass the key as an override, and assert an effect that only the gated path produces.
- A bare `MagicMock` standing in for a DB row or a number fails far from the cause. Return real values.
- supybot's logger has no `%d`; use `%i`. `test_log_format_specifiers.py` enforces it.

## 4. Let the mutants pick the tests

Run the scoped mutation check on each function you are changing before you write its tests. The check lists the survivors, which are the behaviours nothing asserts yet. Write tests from the contract that kill them, then run the check again. This step is done when every survivor in the functions you touched ends up as one of these:

- **killed** by a test;
- **equivalent:** there is no observable difference;
- **cosmetic:** it affects wording, logs or performance only;
- **open:** it's a real gap you are leaving, and you report it.

Give every survivor that isn't killed a one-line reason in your summary.

```
python3 scripts/mutate.py plugins/llm/src/llm/provider.py --wt . \
  --scope _anthropic_image_source --out "${TMPDIR:-/tmp}/mut-provider.json"
python3 -c "import json,sys;[print(r['status'],r['line'],r['desc'],r['src'][:80]) for r in json.load(open(sys.argv[1])) if r['status']!='killed']" "${TMPDIR:-/tmp}/mut-provider.json"
```

- **Choosing scopes.** Repeat `--scope` for more functions. It takes `name` or `Class.method`. A bare name matches every function or class with that name, along with anything nested inside it.
- **Running it.** The check runs in this checkout, one mutant at a time, at about 20 s a mutant. Before mutating, it checks that the module's tests pass unmutated. Don't edit the module or stage files while it runs.
- **If it's killed hard.** The original module is kept at `<out>.orig` until the run ends. After a `kill -9`, copy that file back over the module.
- **Size.** More than about 30 mutants is a job for the cluster sweep, not the laptop.
- **What to chase.** Work only on the code you're changing. A low-severity survivor elsewhere stays where it is.

## 5. Full sweeps

A whole-codebase sweep runs on the OTU cluster with `scripts/mutation_slurm.sh`. The laptop runs `scripts/mutation_sweep.sh`, but its test churn freezes the GUI under Defender.

`docs/superpowers/plans/2026-10-21-mutation-testing-handoff.md` covers the triage → skeptic → test workflow and the last results. Treat an "uncovered" label from a sweep with suspicion: coverage line-mapping mislabelled about 30 mutants that a test in fact killed. Re-run the mutant before writing a test for it.

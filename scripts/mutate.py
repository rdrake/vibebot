"""AST mutation tester, two-tier test selection.

usage: scripts/mutate.py MODULE_REL_PATH --out FILE [--workers N] [--cov cov.json]

Setup and the full-codebase sweep: scripts/mutation_sweep.sh. The work
directory (worktrees wt0..wtN-1, coverage, results) is $MUTATION_DIR,
default ~/.cache/vibebot-mutation.

Tier 1 runs the test files that mention the enclosing function (or class).
A mutant that survives tier 1 is re-run against every test file that imports
the module (tier 2). Mutants on lines no test executes are recorded as
"uncovered" without running anything. Each worker owns worktree wt<i>.
"""

import argparse
import ast
import collections
import concurrent.futures as cf
import json
import os
import pathlib
import re
import subprocess
import time

HERE = pathlib.Path(os.environ.get("MUTATION_DIR", "~/.cache/vibebot-mutation")).expanduser()
TESTS = "plugins"

CMP_SWAP = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}
BIN_SWAP = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.FloorDiv,
}


def docstring_nodes(tree):
    out = set()
    for n in ast.walk(tree):
        if (
            isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and n.body
        ):
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                out.add(id(first.value))
    return out


def enumerate_mutants(tree):
    out = []
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", None)
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in CMP_SWAP:
                    out.append(
                        (
                            node,
                            ln,
                            f"{type(op).__name__}->{CMP_SWAP[type(op)].__name__}",
                            ("cmp", i),
                        )
                    )
        elif isinstance(node, ast.BoolOp):
            out.append(
                (
                    node,
                    ln,
                    f"{type(node.op).__name__}->{'Or' if isinstance(node.op, ast.And) else 'And'}",
                    ("bool",),
                )
            )
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            out.append((node, ln, "drop not", ("not",)))
        elif isinstance(node, ast.BinOp) and type(node.op) in BIN_SWAP:
            out.append(
                (
                    node,
                    ln,
                    f"{type(node.op).__name__}->{BIN_SWAP[type(node.op)].__name__}",
                    ("bin",),
                )
            )
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            out.append((node, ln, f"{node.value}->{node.value + 1}", ("int",)))
        elif isinstance(node, ast.Constant) and type(node.value) is bool:
            out.append((node, ln, f"{node.value}->{not node.value}", ("boolc",)))
        elif isinstance(node, ast.If):
            out.append((node, ln, "if -> False", ("if_false",)))
            out.append((node, ln, "if -> True", ("if_true",)))
        elif (
            isinstance(node, ast.Return)
            and node.value is not None
            and not (isinstance(node.value, ast.Constant) and node.value.value is None)
        ):
            out.append((node, ln, "return -> None", ("ret_none",)))
    return out


def mutant_source(src, index):
    tree = ast.parse(src)
    node, _, _, kind = enumerate_mutants(tree)[index]
    k = kind[0]
    if k == "not":
        for parent in ast.walk(tree):
            for field, val in ast.iter_fields(parent):
                if val is node:
                    setattr(parent, field, node.operand)
                elif isinstance(val, list):
                    for j, v in enumerate(val):
                        if v is node:
                            val[j] = node.operand
    elif k == "cmp":
        node.ops[kind[1]] = CMP_SWAP[type(node.ops[kind[1]])]()
    elif k == "bool":
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif k == "bin":
        node.op = BIN_SWAP[type(node.op)]()
    elif k == "int":
        node.value += 1
    elif k == "boolc":
        node.value = not node.value
    elif k == "if_false":
        node.test = ast.Constant(False)
    elif k == "if_true":
        node.test = ast.Constant(True)
    elif k == "ret_none":
        node.value = ast.Constant(None)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def pytest(wt, tests, timeout):
    try:
        r = subprocess.run(
            [
                # The worktree's own interpreter, not `uv run`: a Homebrew uv
                # upgrade briefly removes the binary and killed a 3-hour run.
                str(wt / ".venv/bin/python"),
                "-m",
                "pytest",
                "-x",
                "-q",
                "-p",
                "no:cacheprovider",
                # One temp root per worker: concurrent runs never share or
                # prune each other's directories.
                f"--basetemp={wt / '.pytest-tmp'}",
                "--no-cov",
                "-m",
                "not slow",
                "-W",
                "ignore",
                *tests,
            ],
            cwd=wt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    except OSError as exc:
        return f"error:{type(exc).__name__}"
    if r.returncode == 0:
        return "survived"
    if r.returncode in (1, 2):
        return "killed"
    if r.returncode == 5:  # no tests collected
        return "survived"
    return f"error{r.returncode}"


def run_one(job):
    wt, module, src, idx, tier1, tier2, timeout = job
    path = wt / module
    path.write_text(mutant_source(src, idx))
    t0 = time.time()
    try:
        st = pytest(wt, tier1, timeout) if tier1 else "survived"
        tier = 1
        if st == "survived" and tier2:
            st = pytest(wt, tier2, timeout)
            tier = 2
    finally:
        path.write_text(src)
    return idx, st, tier, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--cov")
    a = ap.parse_args()
    wts = [HERE / f"wt{i}" for i in range(a.workers)]
    src = (wts[0] / a.module).read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    doc = docstring_nodes(tree)

    executed = None
    if a.cov:
        files = json.loads(pathlib.Path(a.cov).read_text())["files"]
        key = next(
            (k for k in files if k.endswith(a.module.split("src/", 1)[-1]) or k == a.module), None
        )
        executed = set(files[key]["executed_lines"]) if key else set()

    # enclosing function/class name per line, innermost first
    scopes = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scopes.append((n.lineno, n.end_lineno, n.name))

    def scope_names(ln):
        hits = sorted((hi - lo, name) for lo, hi, name in scopes if lo <= ln <= hi)
        return [name for _, name in hits]

    test_files = sorted(str(p.relative_to(wts[0])) for p in (wts[0] / TESTS).rglob("test_*.py"))
    texts = {f: (wts[0] / f).read_text() for f in test_files}
    modname = pathlib.Path(a.module).stem
    pkg_rel = a.module.split("src/", 1)[-1][:-3].replace("/", ".")
    imp = re.compile(rf"\b({re.escape(pkg_rel)}|{re.escape(modname)})\b")
    tier2_all = [f for f in test_files if imp.search(texts[f])] or test_files

    def tier1_for(names):
        for name in names:
            if name.startswith("__"):
                continue
            pat = re.compile(rf"\b{re.escape(name)}\b")
            fs = [f for f in tier2_all if pat.search(texts[f])]
            if fs:
                return fs
        return []

    # A crashed run resumes from its checkpoint instead of starting over.
    partial = pathlib.Path(a.out + ".partial")
    checkpointed = {}
    if partial.exists():
        checkpointed = {r["idx"]: r for r in json.loads(partial.read_text())}

    muts = enumerate_mutants(tree)
    results, jobs = [], []
    for i, (node, ln, desc, _kind) in enumerate(muts):
        if ln is None or id(node) in doc:
            continue
        text = lines[ln - 1]
        s = text.strip()
        if s.startswith("@") or re.search(r"\b(log|logger)\.\w+\(", text) or "__all__" in text:
            continue
        rec = {
            "idx": i,
            "line": ln,
            "desc": desc,
            "scope": ".".join(reversed(scope_names(ln))),
            "src": s[:160],
        }
        if executed is not None and ln not in executed:
            rec.update(status="uncovered", tier=0, secs=0)
            results.append(rec)
            continue
        if i in checkpointed:
            results.append(checkpointed[i])
            continue
        t1 = tier1_for(scope_names(ln))
        t2 = [f for f in tier2_all if f not in t1]
        jobs.append((rec, t1, t2))
    print(
        f"{a.module}: {len(jobs)} to run, {len(checkpointed)} from checkpoint, "
        f"{sum(r['status'] == 'uncovered' for r in results)} uncovered",
        flush=True,
    )

    free = list(wts)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        pending = {}
        it = iter(jobs)

        def submit():
            try:
                rec, t1, t2 = next(it)
            except StopIteration:
                return False
            wt = free.pop()
            f = ex.submit(run_one, (wt, a.module, src, rec["idx"], t1, t2, a.timeout))
            pending[f] = (wt, rec)
            return True

        while free and submit():
            pass
        done_n = 0
        while pending:
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in done:
                wt, rec = pending.pop(f)
                free.append(wt)
                _, st, tier, secs = f.result()
                rec.update(status=st, tier=tier, secs=secs)
                results.append(rec)
                done_n += 1
                if done_n % 50 == 0:
                    partial.write_text(json.dumps([r for r in results if r["tier"]], indent=0))
                if done_n % 100 == 0:
                    print(f"  {done_n}/{len(jobs)}", flush=True)
                submit()
    results.sort(key=lambda r: r["idx"])
    pathlib.Path(a.out).write_text(json.dumps(results, indent=0))
    partial.unlink(missing_ok=True)
    print(a.module, dict(collections.Counter(r["status"] for r in results)), flush=True)


if __name__ == "__main__":
    main()

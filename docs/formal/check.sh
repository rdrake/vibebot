#!/usr/bin/env bash
# Re-run every model and proof in this directory. Needs Java 11+ and, for
# IrcLine.lean, elan (https://github.com/leanprover/elan). TLC's jar is
# fetched into tmp/ (gitignored) on first run.
#
# "expect violation" models are the pre-fix code: they MUST fail, so a pass
# there means the model no longer reproduces the bug it documents.
set -euo pipefail
cd "$(dirname "$0")"
jar=../../tmp/tla2tools.jar
[[ -f $jar ]] || curl -sSfL -o "$jar" https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar

status=0
run() {  # run <module> <pass|violation>
  local out
  out=$(java -XX:+UseParallelGC -cp "$jar" tlc2.TLC -deadlock -workers auto -noGenerateSpecTE \
        -metadir ../../tmp/tlc-states "$1" 2>&1 || true)
  if grep -q "No error has been found" <<<"$out"; then got=pass
  elif grep -q "is violated" <<<"$out"; then got=violation
  else got=error; fi
  if [[ $got == "$2" ]]; then printf 'ok    %-22s %s\n' "$1" "$got"
  else printf 'FAIL  %-22s expected %s, got %s\n' "$1" "$2" "$got"; status=1; fi
}

run TypingHolds         violation
run TypingHoldsSendLock pass
run PTSingle            pass
run PTZombie            violation
run PTZombieFixed       pass

if command -v lean >/dev/null || [[ -x ~/.elan/bin/lean ]]; then
  "$(command -v lean || echo ~/.elan/bin/lean)" IrcLine.lean >/dev/null && echo "ok    IrcLine.lean           proved" \
    || { echo "FAIL  IrcLine.lean"; status=1; }
else
  echo "skip  IrcLine.lean           (no lean on PATH)"
fi
exit $status

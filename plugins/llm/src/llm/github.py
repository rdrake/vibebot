"""Read GitHub links through GitHub's public API instead of Gemini urlContext.

Probed 2026-09-29: urlContext on github.com is unreliable and expensive. A PR
page came back ``URL_RETRIEVAL_STATUS_UNSAFE`` on flash-lite, ``.diff`` URLs
failed on every model, and when it did work it billed ~22k tokens of page
chrome. The bot's worst request in September ("summarize <PR URL>") got empty
fetches back, fell into eight grounded searches, and ran 4.5 minutes.

The API returns the same facts as JSON for a few thousand tokens and no model
call. Pure apart from the ``fetch`` callable, so it tests without a network.
Anything not recognised returns None and the caller keeps its urlContext path.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import quote, urlparse

API = "https://api.github.com"
# Budget for the whole digest. The digest goes straight into the chat model's
# context as a tool result, so this bounds what one link can cost a turn.
MAX_CHARS = 12_000
# Per-file patch excerpt: enough to show what changed, not the whole diff.
MAX_PATCH_CHARS = 1_500
TIMEOUT_SECONDS = 10
# Bytes read from one response. A PR's file list with patches runs to a few
# hundred KB; an 80 MB raw file must not cost 160 MB of RAM to clip to 12 KB.
# JSON cut off here fails to parse, and the digest falls back to urlContext.
MAX_FETCH_BYTES = 2_000_000

Fetch = Callable[[str, bool], str]


def http_fetch(url: str, raw: bool = False) -> str:
    """GET ``url``; JSON text by default, the raw body when ``raw``.

    Unauthenticated: prod holds no GitHub token, and 60 requests/hour per IP is
    far above what an IRC channel pastes.
    """
    accept = "application/vnd.github.raw" if raw else "application/vnd.github+json"
    req = urllib.request.Request(url, headers={"User-Agent": "VibeBot/8", "Accept": accept})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:  # noqa: S310 — fixed hosts
        return resp.read(MAX_FETCH_BYTES).decode("utf-8", errors="replace")


_REPO = r"/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)"
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("pull", re.compile(_REPO + r"/pull/(?P<num>\d+)")),
    ("issue", re.compile(_REPO + r"/issues/(?P<num>\d+)")),
    ("commit", re.compile(_REPO + r"/commit/(?P<sha>[0-9a-fA-F]{4,40})")),
    ("blob", re.compile(_REPO + r"/blob/(?P<ref>[^/]+)/(?P<path>.+)")),
    ("repo", re.compile(_REPO + r"/?$")),
]


def parse(url: str) -> tuple[str, dict[str, str]] | None:
    """Classify a GitHub URL as (kind, fields), or None if it is not one we read."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    path = parsed.path.removesuffix(".diff").removesuffix(".patch")
    if host in ("raw.githubusercontent.com", "gist.githubusercontent.com"):
        return "raw", {"url": url}
    if host == "gist.github.com":
        m = re.match(r"/(?:[\w.-]+/)?(?P<id>[0-9a-fA-F]+)/?$", path)
        return ("gist", m.groupdict()) if m else None
    if host not in ("github.com", "www.github.com"):
        return None
    for kind, pattern in _PATTERNS:
        m = pattern.match(path)
        if m:
            return kind, m.groupdict()
    return None


def _clip(text: str | None, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n[…truncated]"


def _files(files: list[dict], budget: int) -> list[str]:
    """One line per changed file, then patch excerpts until the budget runs out."""
    lines = [
        f"- {f.get('filename')} ({f.get('status')}, +{f.get('additions', 0)}/-{f.get('deletions', 0)})"
        for f in files
    ]
    for f in files:
        patch = f.get("patch")
        if not patch or budget <= 0:
            continue
        excerpt = _clip(patch, min(MAX_PATCH_CHARS, budget))
        lines.append(f"\n--- {f.get('filename')}\n{excerpt}")
        budget -= len(excerpt)
    return lines


def _repo_base(f: dict[str, str]) -> str:
    return f"{API}/repos/{quote(f['owner'])}/{quote(f['repo'])}"


def digest(url: str, fetch: Fetch = http_fetch) -> str | None:
    """Plain-text digest of a GitHub URL, or None to fall back to urlContext.

    None on any failure — rate limit, 404, private repo, odd JSON — because the
    caller's generic path may still manage, and a GitHub hiccup is not worth an
    error in the channel.
    """
    target = parse(url)
    if target is None:
        return None
    kind, f = target
    try:
        out = _build(kind, f, fetch)
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError, OSError):
        return None
    if not out:
        return None
    return _clip(f"Source: {url}\n{out}", MAX_CHARS)


def _build(kind: str, f: dict[str, str], fetch: Fetch) -> str:
    if kind == "raw":
        return _clip(fetch(f["url"], True), MAX_CHARS)

    if kind == "gist":
        g = json.loads(fetch(f"{API}/gists/{quote(f['id'])}", False))
        parts = [
            f"Gist by {(g.get('owner') or {}).get('login', '?')}: {g.get('description') or ''}"
        ]
        for name, file in (g.get("files") or {}).items():
            parts.append(f"\n--- {name}\n{_clip(file.get('content'), MAX_CHARS // 2)}")
        return "\n".join(parts)

    base = _repo_base(f)
    if kind == "repo":
        r = json.loads(fetch(base, False))
        head = (
            f"Repository {r.get('full_name')}: {r.get('description') or ''}\n"
            f"Language: {r.get('language')} · Stars: {r.get('stargazers_count')} · "
            f"Updated: {r.get('pushed_at')}"
        )
        try:
            readme = fetch(f"{base}/readme", True)
        except urllib.error.HTTPError:
            readme = ""
        return head + (f"\n\nREADME:\n{_clip(readme, MAX_CHARS // 2)}" if readme else "")

    if kind == "blob":
        text = fetch(f"{base}/contents/{quote(f['path'])}?ref={quote(f['ref'])}", True)
        return f"File {f['path']} @ {f['ref']}:\n{_clip(text, MAX_CHARS)}"

    if kind == "commit":
        c = json.loads(fetch(f"{base}/commits/{quote(f['sha'])}", False))
        meta = c.get("commit") or {}
        stats = c.get("stats") or {}
        head = (
            f"Commit {str(c.get('sha', ''))[:10]} by {(meta.get('author') or {}).get('name')} "
            f"on {(meta.get('author') or {}).get('date')} "
            f"(+{stats.get('additions', 0)}/-{stats.get('deletions', 0)})\n\n"
            f"{_clip(meta.get('message'), 3_000)}\n\nFiles:"
        )
        return "\n".join([head, *_files(c.get("files") or [], MAX_CHARS - len(head) - 500)])

    if kind == "pull":
        p = json.loads(fetch(f"{base}/pulls/{f['num']}", False))
        state = "merged" if p.get("merged") else p.get("state")
        head = (
            f"Pull request #{f['num']}: {p.get('title')}\n"
            f"By {(p.get('user') or {}).get('login')} · {state} · "
            f"{p.get('head', {}).get('label')} → {p.get('base', {}).get('label')} · "
            f"{p.get('commits')} commits, +{p.get('additions', 0)}/-{p.get('deletions', 0)} "
            f"in {p.get('changed_files')} files\n\n{_clip(p.get('body'), 3_000)}\n\nFiles:"
        )
        files = json.loads(fetch(f"{base}/pulls/{f['num']}/files?per_page=100", False))
        return "\n".join([head, *_files(files, MAX_CHARS - len(head) - 500)])

    if kind == "issue":
        i = json.loads(fetch(f"{base}/issues/{f['num']}", False))
        head = (
            f"Issue #{f['num']}: {i.get('title')}\n"
            f"By {(i.get('user') or {}).get('login')} · {i.get('state')} · "
            f"{i.get('comments', 0)} comments\n\n{_clip(i.get('body'), 3_000)}"
        )
        comments = json.loads(fetch(f"{base}/issues/{f['num']}/comments?per_page=20", False))
        parts = [head]
        budget = MAX_CHARS - len(head) - 500
        for cm in comments:
            line = f"\n{(cm.get('user') or {}).get('login')}: {_clip(cm.get('body'), 800)}"
            if len(line) > budget:
                break
            parts.append(line)
            budget -= len(line)
        return "\n".join(parts)

    return ""

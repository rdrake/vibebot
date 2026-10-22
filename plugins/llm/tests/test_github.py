"""GitHub link digests: URL classification, formatting, budget, and fallback."""

from __future__ import annotations

import base64
import json
import urllib.error

import pytest
from llm import github


class TestParse:
    @pytest.mark.parametrize(
        ("url", "kind"),
        [
            ("https://github.com/evilnet/nefarious2/pull/107", "pull"),
            ("https://github.com/evilnet/nefarious2/pull/107/files", "pull"),
            ("https://github.com/evilnet/nefarious2/pull/107.diff", "pull"),
            ("https://github.com/evilnet/nefarious2/pull/107#pullrequestreview-5", "pull"),
            ("https://github.com/o/r/issues/12", "issue"),
            ("https://github.com/o/r/commit/2d0fdec175a1", "commit"),
            ("https://github.com/o/r/commit/2d0fdec175a1.patch", "commit"),
            ("https://github.com/o/r/blob/main/src/a.c", "blob"),
            ("https://github.com/rdrake/vibebot-v8", "repo"),
            ("https://github.com/rdrake/vibebot-v8/", "repo"),
            ("https://gist.github.com/MrLenin/491f87ea4d95", "gist"),
            ("https://gist.githubusercontent.com/MrLenin/491f/raw/x.md", "raw"),
            ("https://raw.githubusercontent.com/o/r/main/README.md", "raw"),
        ],
    )
    def test_recognised(self, url: str, kind: str) -> None:
        parsed = github.parse(url)
        assert parsed is not None
        assert parsed[0] == kind

    @pytest.mark.parametrize(
        "url",
        [
            "https://rdrake.github.io/vibebot-v8/",
            "https://www.githubstatus.com/incidents/x",
            "https://github.com/o/r/security/advisories/GHSA-1",
            "https://github.com/o/r/actions/runs/1",
            "https://example.com/o/r/pull/1",
            "https://evil.example/github.com/o/r/pull/1",
        ],
    )
    def test_not_recognised(self, url: str) -> None:
        assert github.parse(url) is None
        assert github.digest(url, fetch=_never) is None


def _never(url: str, raw: bool) -> str:
    raise AssertionError(f"unexpected fetch {url}")


class _RawFile:
    """A contents-API body that answers the way GitHub does for each Accept.

    ``application/vnd.github.raw`` gets the file text; the default JSON Accept
    gets a metadata envelope with the text base64-encoded, which is no use in a
    digest.
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def serve(self, raw: bool) -> str:
        if raw:
            return self.text
        encoded = base64.b64encode(self.text.encode()).decode()
        return json.dumps({"type": "file", "encoding": "base64", "content": encoded})


def _fake(routes: dict[str, object]):
    calls: list[str] = []

    def fetch(url: str, raw: bool) -> str:
        calls.append(url)
        body = routes[url]
        if isinstance(body, Exception):
            raise body
        if isinstance(body, _RawFile):
            return body.serve(raw)
        return body if isinstance(body, str) else json.dumps(body)

    return fetch, calls


BASE = "https://api.github.com/repos/o/r"


class TestDigest:
    def test_pull_request(self) -> None:
        fetch, calls = _fake(
            {
                f"{BASE}/pulls/7": {
                    "title": "Fix push",
                    "user": {"login": "sean"},
                    "merged": True,
                    "state": "closed",
                    "head": {"label": "sean:fix"},
                    "base": {"label": "o:main"},
                    "commits": 1,
                    "additions": 3,
                    "deletions": 1,
                    "changed_files": 1,
                    "body": "Explains the change.",
                },
                f"{BASE}/pulls/7/files?per_page=100": [
                    {
                        "filename": "a.c",
                        "status": "modified",
                        "additions": 3,
                        "deletions": 1,
                        "patch": "@@ -1 +1 @@\n-old\n+new",
                    },
                ],
            }
        )
        text = github.digest("https://github.com/o/r/pull/7", fetch=fetch)
        assert text is not None
        assert "Pull request #7: Fix push" in text
        assert "merged" in text
        assert "Explains the change." in text
        assert "- a.c (modified, +3/-1)" in text
        assert "+new" in text
        assert len(calls) == 2

    def test_commit(self) -> None:
        fetch, _ = _fake(
            {
                f"{BASE}/commits/abc123": {
                    "sha": "abc1234567890",
                    "commit": {
                        "message": "Do the thing",
                        "author": {"name": "Ann", "date": "2026-09-01"},
                    },
                    "stats": {"additions": 1, "deletions": 0},
                    "files": [
                        {"filename": "b.py", "status": "added", "additions": 1, "deletions": 0}
                    ],
                }
            }
        )
        text = github.digest("https://github.com/o/r/commit/abc123", fetch=fetch)
        assert text is not None
        assert "Commit abc1234567 by Ann" in text
        # The head carries the commit's own totals from the API's stats object.
        head_line = next(ln for ln in text.splitlines() if ln.startswith("Commit "))
        assert "(+1/-0)" in head_line
        assert "Do the thing" in text
        assert "- b.py (added, +1/-0)" in text

    def test_issue_with_comments(self) -> None:
        fetch, _ = _fake(
            {
                f"{BASE}/issues/3": {
                    "title": "Crash",
                    "user": {"login": "u"},
                    "state": "open",
                    "comments": 1,
                    "body": "It crashes.",
                },
                f"{BASE}/issues/3/comments?per_page=20": [
                    {"user": {"login": "m"}, "body": "Repro?"}
                ],
            }
        )
        text = github.digest("https://github.com/o/r/issues/3", fetch=fetch)
        assert text is not None
        assert "Issue #3: Crash" in text
        assert "m: Repro?" in text

    def test_repo_without_readme_still_digests(self) -> None:
        fetch, _ = _fake(
            {
                BASE: {
                    "full_name": "o/r",
                    "description": "A bot",
                    "language": "Python",
                    "stargazers_count": 5,
                    "pushed_at": "2026-09-28",
                },
                f"{BASE}/readme": urllib.error.HTTPError(f"{BASE}/readme", 404, "nf", None, None),  # type: ignore[arg-type]
            }
        )
        text = github.digest("https://github.com/o/r", fetch=fetch)
        assert text is not None
        assert "Repository o/r: A bot" in text
        assert "README" not in text

    def test_repo_includes_readme_text(self) -> None:
        fetch, _ = _fake(
            {
                BASE: {"full_name": "o/r", "description": "A bot"},
                f"{BASE}/readme": _RawFile("# Title\nWelcome aboard."),
            }
        )
        text = github.digest("https://github.com/o/r", fetch=fetch)
        assert text is not None
        assert "README:\n# Title\nWelcome aboard." in text

    def test_blob_shows_the_file_text_at_its_ref(self) -> None:
        fetch, _ = _fake({f"{BASE}/contents/src/a.c?ref=main": _RawFile("int answer = 42;")})
        text = github.digest("https://github.com/o/r/blob/main/src/a.c", fetch=fetch)
        assert text is not None
        assert "File src/a.c @ main:\nint answer = 42;" in text

    def test_gist_shows_owner_description_and_each_file(self) -> None:
        fetch, _ = _fake(
            {
                "https://api.github.com/gists/491f": {
                    "owner": {"login": "MrLenin"},
                    "description": "notes on push",
                    "files": {
                        "a.md": {"content": "first file body"},
                        "b.sh": {"content": "echo second"},
                    },
                }
            }
        )
        text = github.digest("https://gist.github.com/MrLenin/491f", fetch=fetch)
        assert text is not None
        assert "Gist by MrLenin: notes on push" in text
        assert "--- a.md\nfirst file body" in text
        assert "--- b.sh\necho second" in text

    def test_raw_is_fetched_verbatim(self) -> None:
        url = "https://raw.githubusercontent.com/o/r/main/x.txt"
        fetch, _ = _fake({url: "hello"})
        assert github.digest(url, fetch=fetch) == f"Source: {url}\nhello"

    def test_digest_is_capped(self) -> None:
        huge = [
            {
                "filename": f"f{i}.c",
                "status": "modified",
                "additions": 1,
                "deletions": 1,
                "patch": "x" * 5_000,
            }
            for i in range(50)
        ]
        fetch, _ = _fake(
            {
                f"{BASE}/pulls/1": {"title": "t", "body": "b" * 10_000},
                f"{BASE}/pulls/1/files?per_page=100": huge,
            }
        )
        text = github.digest("https://github.com/o/r/pull/1", fetch=fetch)
        assert text is not None
        assert len(text) <= github.MAX_CHARS + len("\n[…truncated]")

    @pytest.mark.parametrize(
        "error",
        [
            urllib.error.HTTPError("u", 403, "rate limited", None, None),  # type: ignore[arg-type]
            urllib.error.URLError("dns"),
            TimeoutError(),
            ValueError("bad json"),
        ],
    )
    def test_failures_fall_back_with_none(self, error: Exception) -> None:
        fetch, _ = _fake({f"{BASE}/pulls/1": error})
        assert github.digest("https://github.com/o/r/pull/1", fetch=fetch) is None


class _Body:
    """urlopen stand-in that serves ``data`` and records each read size."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.reads: list[int | None] = []

    def read(self, amount: int | None = None) -> bytes:
        self.reads.append(amount)
        return self.data if amount is None else self.data[:amount]

    def __enter__(self) -> _Body:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class TestHttpFetchIsBounded:
    def test_huge_raw_file_is_read_only_up_to_the_cap(self, mocker) -> None:
        """GIVEN an 80 MB raw file WHEN fetched THEN only the cap is read, not the body."""
        body = _Body(b"x" * (github.MAX_FETCH_BYTES * 4))
        mocker.patch("urllib.request.urlopen", return_value=body)

        text = github.http_fetch("https://raw.githubusercontent.com/o/r/main/big.txt", True)

        assert body.reads == [github.MAX_FETCH_BYTES]
        assert len(text) == github.MAX_FETCH_BYTES

    def test_oversized_json_falls_back_with_none(self, mocker) -> None:
        """GIVEN JSON past the cap WHEN digested THEN the truncated text fails
        to parse and the caller keeps its urlContext path."""
        huge = json.dumps({"title": "t", "body": "b" * github.MAX_FETCH_BYTES}).encode()
        mocker.patch("urllib.request.urlopen", return_value=_Body(huge))

        assert github.digest("https://github.com/o/r/pull/1") is None


class TestUrlCompletionUsesDigest:
    def test_github_link_skips_the_model(self, make_service, mocker) -> None:
        service, _ = make_service()
        mocker.patch("llm.service.github.digest", return_value="Source: x\nPR body")
        grounded = mocker.patch.object(service, "_grounded_completion")
        result = service.url_completion("https://github.com/o/r/pull/1", channel="#t")
        grounded.assert_not_called()
        assert "<fetched_content>\nSource: x\nPR body\n</fetched_content>" in result.content
        assert result.cost == 0.0

    def test_other_links_keep_url_context(self, make_service, mocker) -> None:
        service, _ = make_service()
        mocker.patch("llm.service.github.digest", return_value=None)
        grounded = mocker.patch.object(service, "_grounded_completion")
        service.url_completion("https://example.com/page", channel="#t")
        grounded.assert_called_once()

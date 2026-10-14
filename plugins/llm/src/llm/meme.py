"""Meme templates: request parsing, the bundled catalog, and name resolution.

The user names a template — "drake", "distracted boyfriend" — and the
catalog resolves that name deterministically; :mod:`meme_render` draws the
captions. The catalog is ``memes/templates.json`` beside this file, one
entry per template with its caption boxes, built by
``scripts/build_meme_catalog.py`` from memegen's hand-made layouts and
imgflip's blanks with vision-drafted boxes.

Two callers share it: the ``@meme`` command (no LLM at all) and the
``make_meme`` chat tool, where the model's job is to transcribe the name the
user said, not to pick one. Both go through :func:`plan_meme`, so a wrong
caption count or an unknown name reads the same in either place.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .meme_render import FONT_FILES, TOP_BOTTOM, Box, Slot
from .meme_topics import PICTURE_SLOT

# A caption past this length renders as unreadable four-point text anyway.
MAX_CAPTION_CHARS = 200
MAX_LINES = 8

CATALOG_PATH = Path(__file__).parent / "memes" / "templates.json"

_NOISE_WORDS = frozenset({"the", "a", "an", "meme", "template", "of", "and"})
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# The bring-your-own-image template: a top and a bottom band over a pasted
# image URL or a URL-valued alias.
CUSTOM_TEMPLATE_ID = "custom"
CUSTOM_LINES = len(TOP_BOTTOM)


@dataclass(frozen=True)
class MemeTemplate:
    id: str
    name: str
    lines: int
    keywords: tuple[str, ...] = ()
    example: tuple[str, ...] = ()
    # Set only on custom templates: the image URL the captions go on.
    background: str | None = None
    # One per caption, in caption order. Custom templates use TOP_BOTTOM.
    boxes: tuple[Box, ...] = ()
    # The blank, for bundled templates.
    image: Path | None = None
    # Popularity order (imgflip's); lower is more popular.
    rank: int = 10_000
    # Blank picture areas a generated picture is masked into, in order.
    slots: tuple[Slot, ...] = ()


FONTS = tuple(FONT_FILES)


@dataclass(frozen=True)
class MemeOptions:
    """How the user asked for the meme beyond template and captions."""

    font: str | None = None
    layout_top: bool = False
    # An instruction for the image-edit model, applied to the captioned
    # meme — what goes in the Spirit Halloween costume's blank photo, say.
    # Written by the picker from the request, or by the chat model.
    draw: str | None = None


def custom_template(background: str, name: str = "custom image") -> MemeTemplate:
    return MemeTemplate(
        CUSTOM_TEMPLATE_ID, name, CUSTOM_LINES, background=background, boxes=TOP_BOTTOM
    )


def is_http_url(text: str) -> bool:
    return text.lower().startswith(("http://", "https://")) and " " not in text


def _safe_background(url: str) -> bool:
    from .service import validate_external_url

    return is_http_url(url) and validate_external_url(url)


@dataclass(frozen=True)
class MemePlan:
    """What :func:`plan_meme` decided: the captions to draw, or why not.

    ``lines`` is padded with blanks to the template's box count.
    """

    lines: list[str] | None
    template: MemeTemplate | None
    error: str | None


def parse_meme_request(text: str) -> tuple[str, list[str]] | None:
    """Split ``template | line | line`` into its parts.

    Returns None when there is no template name; a template with no lines is
    a valid request (plan_meme answers it with the template's line count and
    example) so ``@meme drake`` teaches the format.
    """
    parts = [p.strip() for p in text.split("|")]
    if not parts or not parts[0]:
        return None
    return parts[0], parts[1:]


def parse_catalog(entries: Any, image_dir: Path) -> list[MemeTemplate]:
    """Read ``templates.json``. Malformed entries are dropped, not raised."""
    templates: list[MemeTemplate] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        template_id = entry.get("id")
        name = entry.get("name")
        if not (isinstance(template_id, str) and template_id and isinstance(name, str)):
            continue
        boxes: list[Box] = []
        for raw in entry.get("boxes") or []:
            try:
                boxes.append(Box(**raw))
            except TypeError:
                continue
        if not boxes:
            continue
        keywords = tuple(k for k in entry.get("keywords") or [] if isinstance(k, str))
        example = tuple(t for t in entry.get("example") or [] if isinstance(t, str))
        rank = entry.get("rank")
        slots: list[Slot] = []
        for raw in entry.get("slots") or []:
            try:
                x, y, w, h = (float(v) for v in raw["box"])
                mask = image_dir.parent / "masks" / str(raw["mask"])
                corners = tuple((float(c[0]), float(c[1])) for c in raw.get("corners") or [])
                slots.append(Slot((x, y, w, h), mask, corners if len(corners) == 4 else ()))
            except (KeyError, TypeError, ValueError):
                continue
        if slots and PICTURE_SLOT not in keywords:
            keywords = (PICTURE_SLOT, *keywords)
        templates.append(
            MemeTemplate(
                template_id,
                name,
                len(boxes),
                keywords,
                example,
                boxes=tuple(boxes),
                image=image_dir / f"{template_id}.jpg",
                rank=rank if isinstance(rank, int) else 10_000,
                slots=tuple(slots),
            )
        )
    return templates


_catalog_lock = threading.Lock()
_catalog_cache: tuple[Path, float, MemeCatalog] | None = None


def bundled_catalog(path: Path = CATALOG_PATH) -> MemeCatalog | None:
    """The catalog in ``path``, re-read when the file changes. None if missing."""
    global _catalog_cache
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    with _catalog_lock:
        if _catalog_cache is None or _catalog_cache[:2] != (path, mtime):
            entries = json.loads(path.read_text(encoding="utf-8"))
            catalog = MemeCatalog(parse_catalog(entries, path.parent / "images"))
            _catalog_cache = (path, mtime, catalog)
        return _catalog_cache[2]


def _normalise(text: str) -> str:
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _NOISE_WORDS]
    return " ".join(tokens)


def _haystack(template: MemeTemplate) -> str:
    return _normalise(f"{template.id} {template.name} {' '.join(template.keywords)}")


def _search_text(template: MemeTemplate) -> str:
    """The haystack plus the example captions — what ``list`` and the
    suggestions search. Kept out of :meth:`MemeCatalog.resolve`: "workout"
    should find Butthurt Dweller in a listing, not caption it unasked."""
    return _normalise(f"{_haystack(template)} {' '.join(template.example)}")


def _overlap(query_tokens: list[str], template: MemeTemplate) -> tuple[int, int]:
    """``(words shared with the id or name, words shared with anything)``.

    The name counts first so the topic tags cannot outvote it: "elmo fire"
    must still reach Elmo when three other templates are tagged "fire".
    """
    named = set(_normalise(f"{template.id} {template.name}").split())
    words = set(_haystack(template).split())
    return (
        sum(1 for tok in query_tokens if tok in named),
        sum(1 for tok in query_tokens if tok in words),
    )


class MemeCatalog:
    """The template list plus deterministic name resolution."""

    def __init__(self, templates: list[MemeTemplate]) -> None:
        self.templates = list(templates)
        self._by_id = {t.id.lower(): t for t in self.templates}

    def __len__(self) -> int:
        return len(self.templates)

    def with_keywords(self, extra: Mapping[str, Iterable[str]]) -> MemeCatalog:
        """A copy with more keywords on the named templates.

        Ids the catalog does not have are ignored: the topic table in
        :mod:`meme_topics` outlives any one catalog build.
        """
        merged = [
            replace(t, keywords=(*t.keywords, *extra[t.id.lower()])) if t.id.lower() in extra else t
            for t in self.templates
        ]
        return MemeCatalog(merged)

    def with_aliases(self, aliases: dict[str, str]) -> MemeCatalog:
        """A copy with operator aliases folded in.

        ``name=id`` makes ``name`` a keyword on that template, so both
        resolve and ``@meme list`` see it. ``name=https://...`` adds a two-box
        custom template drawn on that image. Aliases pointing at an unknown
        id or an unsafe URL are dropped, not raised: a typo in bot.conf must
        not take every meme down with it.
        """
        extra_keywords: dict[str, list[str]] = {}
        customs: list[MemeTemplate] = []
        for name, target in aliases.items():
            if is_http_url(target):
                if _safe_background(target):
                    customs.append(
                        MemeTemplate(
                            _normalise(name).replace(" ", "-") or name,
                            name,
                            CUSTOM_LINES,
                            background=target,
                            boxes=TOP_BOTTOM,
                            rank=0,
                        )
                    )
            elif target.lower() in self._by_id:
                extra_keywords.setdefault(target.lower(), []).append(name)
        return MemeCatalog(self.with_keywords(extra_keywords).templates + customs)

    def exact(self, query: str) -> MemeTemplate | None:
        """The template whose id or name IS ``query`` — no fuzzy tiers.

        The command uses this to tell "@meme drake" (teach the format) from
        "@meme y'all got any more of them flamethrowers", which the overlap
        tier would also resolve to a template and which is a request, not a
        name.
        """
        q = query.strip().lower()
        if q in self._by_id:
            return self._by_id[q]
        nq = _normalise(q)
        if nq in self._by_id:
            return self._by_id[nq]
        by_name = [t for t in self.templates if _normalise(t.name) == nq]
        return by_name[0] if len(by_name) == 1 else None

    def resolve(self, query: str) -> MemeTemplate | None:
        """Exact id, exact name, UNIQUE name substring, keyword, then word overlap.

        Ambiguity is a miss on purpose at every tier: "cat" matching Grumpy
        Cat and Business Cat must not silently pick one, and "expanding
        brain" tying Galaxy Brain with Scumbag Brain must not either. The
        caller lists the candidates instead.
        """
        q = query.strip().lower()
        if not q:
            return None
        if q in self._by_id:
            return self._by_id[q]
        nq = _normalise(q)
        if not nq:
            return None
        if nq in self._by_id:
            return self._by_id[nq]
        by_name = [t for t in self.templates if _normalise(t.name) == nq]
        if len(by_name) == 1:
            return by_name[0]
        by_substring = [t for t in self.templates if nq in _normalise(t.name)]
        if len(by_substring) == 1:
            return by_substring[0]
        if len(by_substring) > 1:
            return None
        by_keyword = [t for t in self.templates if any(nq == _normalise(k) for k in t.keywords)]
        if len(by_keyword) == 1:
            return by_keyword[0]
        if by_keyword:
            return None
        # Templates are often named by the quote; people name them by the
        # character. "willy wonka" has to reach Condescending Wonka on the
        # one word they share, but one word of five is a coincidence.
        tokens = nq.split()
        scored = sorted(((_overlap(tokens, t), t) for t in self.templates), key=lambda s: s[0])
        scored.reverse()
        best, winner = scored[0]
        if best[1] == 0 or best[1] * 2 < len(tokens):
            return None
        if len(scored) > 1 and scored[1][0] == best:
            return None
        return winner

    def suggest(self, query: str, limit: int = 5) -> list[MemeTemplate]:
        """Closest names first. Empty when nothing shares a word with the
        query — an alphabetical head would only look like advice."""
        nq = _normalise(query)
        if not nq:
            return self.templates[:limit]
        # "push-up" must not list every "shut up" and "wake up": a word this
        # short only counts when it is the whole query.
        tokens = [tok for tok in nq.split() if len(tok) >= 3] or nq.split()
        scored: list[tuple[int, int, int, str, MemeTemplate]] = []
        for t in self.templates:
            haystack = _search_text(t)
            words = set(haystack.split())
            pos = haystack.find(nq)
            overlap = sum(1 for tok in tokens if tok in words)
            if pos < 0 and overlap == 0:
                continue
            scored.append((0 if pos >= 0 else 1, -overlap, len(t.name), t.id, t))
        scored.sort()
        return [s[4] for s in scored[:limit]]


# One catalog line per template for the picker prompt. imgflip alt names
# and drafted keywords run to whole sentences ("corporate needs you to find
# the differences between this picture and this picture"); at eight of
# those a row averaged 270 chars and the brief 63 KB. Short tags only, five
# of them, and a clipped example keep it near 35 KB.
_BRIEF_TAGS = 5
_BRIEF_TAG_WORDS = 3
_BRIEF_EXAMPLE_CHARS = 60


def _brief_tags(keywords: Iterable[str], name: str = "") -> str:
    """Up to five short, distinct tags that are not just the name; the
    picture-slot tag always first."""
    keywords = list(keywords)
    in_name = _normalise(name)
    ordered = ([PICTURE_SLOT] if PICTURE_SLOT in keywords else []) + [
        k for k in keywords if k != PICTURE_SLOT
    ]
    seen: set[str] = set()
    tags: list[str] = []
    for k in ordered:
        key = _normalise(k)
        if not key or key in seen or len(k.split()) > _BRIEF_TAG_WORDS or key in in_name:
            continue
        seen.add(key)
        tags.append(k)
        if len(tags) == _BRIEF_TAGS:
            break
    return ", ".join(tags)


def _brief_example(example: tuple[str, ...]) -> str:
    text = " / ".join(x or "_" for x in example)
    if len(text) <= _BRIEF_EXAMPLE_CHARS:
        return text
    return text[: _BRIEF_EXAMPLE_CHARS - 1].rstrip() + "…"


# The whole catalog is over a thousand rows; the picker sees the most
# popular templates plus the ones that share a word with the request, which
# keeps the brief near the ~30 KB it was with memegen's two hundred.
BRIEF_POPULAR = 150
BRIEF_MATCHES = 60
# With the search_templates tool the opening brief only has to cover the
# obvious picks; anything else the picker looks up.
SEARCH_BRIEF_POPULAR = 40
SEARCH_BRIEF_MATCHES = 20
SEARCH_RESULTS = 15


def brief_templates(
    catalog: MemeCatalog,
    request: str = "",
    *,
    popular: int | None = None,
    matches: int | None = None,
) -> list[MemeTemplate]:
    """The shortlist :func:`catalog_brief` shows: request matches, custom
    (URL-alias) templates, then the most popular."""
    popular = BRIEF_POPULAR if popular is None else popular
    matches = BRIEF_MATCHES if matches is None else matches
    chosen: dict[str, MemeTemplate] = {}
    if request.strip() and matches:
        for t in catalog.suggest(request, limit=matches):
            chosen.setdefault(t.id, t)
    for t in catalog.templates:
        if t.background:
            chosen.setdefault(t.id, t)
    for t in sorted(catalog.templates, key=lambda t: t.rank)[:popular]:
        chosen.setdefault(t.id, t)
    return list(chosen.values())


def _brief_row(t: MemeTemplate) -> str:
    example = _brief_example(t.example) if t.example else ""
    tags = _brief_tags(t.keywords, t.name)
    if len(t.slots) > 1:
        # The picker writes one picture per area, separated by " | ".
        tags = tags.replace(PICTURE_SLOT, f"{PICTURE_SLOT} x{len(t.slots)}", 1)
    return f"{t.id} | {t.name} | {t.lines} | {example} | {tags}"


# The aspect ratios xAI's image generation takes.
ASPECT_RATIOS = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16")


def nearest_aspect_ratio(width: int, height: int) -> str:
    """The supported ratio closest to ``width:height`` (compared as logs,
    so 2:1 and 1:2 are equally far from 1:1)."""
    import math

    target = math.log(width / height)

    def distance(ratio: str) -> float:
        w, h = (int(x) for x in ratio.split(":"))
        return abs(math.log(w / h) - target)

    return min(ASPECT_RATIOS, key=distance)


def slot_drawings(draw: str, count: int) -> list[str]:
    """One picture description per slot from ``a | b | c``.

    Fewer descriptions than slots repeat the last one — "same" wants the
    same picture twice, and a one-line draw on a two-slot template should
    still fill both.
    """
    parts = [p.strip() for p in draw.split(" | ") if p.strip()] or [draw.strip()]
    return [parts[min(i, len(parts) - 1)] for i in range(count)]


def catalog_brief(
    catalog: MemeCatalog,
    request: str = "",
    *,
    popular: int | None = None,
    matches: int | None = None,
) -> str:
    """The shortlist as the picker sees it: ``id | name | N | example | tags``."""
    templates = brief_templates(catalog, request, popular=popular, matches=matches)
    return "\n".join(_brief_row(t) for t in templates)


def search_brief(catalog: MemeCatalog, query: str, limit: int = SEARCH_RESULTS) -> str:
    """The picker's ``search_templates`` answer: matching rows, best first.

    The same word matching as ``@meme list`` — names, alt names, tags and
    example captions — so a search for a character or a situation finds
    what a person listing templates would.
    """
    found = catalog.suggest(query, limit=limit) if query.strip() else []
    if not found:
        return f"No templates match '{query.strip()}'. Try other words, or pick from the catalog."
    return "\n".join(_brief_row(t) for t in found)


@dataclass(frozen=True)
class MemeChoice:
    template: MemeTemplate
    lines: list[str]
    draw: str | None = None


def parse_pick(content: str | None, catalog: MemeCatalog) -> MemeChoice | str:
    """Validate the picker's answer against the catalog; a string is why not.

    The model's JSON is ``{"template": id, "lines": [...]}`` or
    ``{"template": null, "reason": ...}``. Anything but an id from the
    catalog is a miss — the whole point of the picker running inside the
    tool is that an invented name never reaches the renderer. Too many lines are
    cut, not refused: the model miscounting boxes is not the user's problem.
    """
    from .service import _extract_json_object

    parsed = _extract_json_object(content)
    if not isinstance(parsed, dict):
        return "The picker did not answer."
    template_id = parsed.get("template")
    if not isinstance(template_id, str) or not template_id.strip():
        reason = parsed.get("reason")
        return (
            str(reason).strip() if isinstance(reason, str) and reason.strip() else "Nothing fits."
        )
    template = catalog.resolve(template_id)
    if template is None:
        return f"The picker chose '{template_id}', which is not a template."
    raw_lines = parsed.get("lines")
    if not isinstance(raw_lines, list):
        return "The picker gave no captions."
    lines = [str(x).strip()[:MAX_CAPTION_CHARS] for x in raw_lines][: template.lines]
    if not any(lines):
        return f"The picker chose {template.id} but wrote no captions."
    draw = parsed.get("draw")
    draw = draw.strip()[:MAX_CAPTION_CHARS] if isinstance(draw, str) and draw.strip() else None
    return MemeChoice(template, lines, draw)


def parse_aliases(entries: list[str]) -> dict[str, str]:
    """``name=id`` or ``name=https://image`` pairs from memeAliases."""
    aliases: dict[str, str] = {}
    for entry in entries or []:
        name, sep, target = str(entry).partition("=")
        name, target = name.strip(), target.strip()
        if sep and name and target:
            aliases[name] = target
    return aliases


def describe_short(template: MemeTemplate) -> str:
    """``id (Name, lines)`` — the @meme list line."""
    return f"{template.id} ({template.name}, {template.lines})"


def _describe(template: MemeTemplate) -> str:
    example = " | ".join(template.example) if template.example else ""
    tail = f" — e.g. {template.id} | {example}" if example else ""
    plural = "caption" if template.lines == 1 else "captions"
    return f"{template.id} ({template.name}) takes {template.lines} {plural}{tail}"


def plan_meme(
    catalog: MemeCatalog,
    template_query: str,
    lines: list[str],
    options: MemeOptions | None = None,
) -> MemePlan:
    """Resolve the name and check the captions and options.

    Fewer captions than boxes is fine (the rest render blank — most people
    only want two on a three-box template); more is not, since there is no
    box to put them in.
    """
    options = options or MemeOptions()
    if is_http_url(template_query.strip()):
        background = template_query.strip()
        if not _safe_background(background):
            return MemePlan(
                None, None, "That image URL is not one I can use (http(s), public host)."
            )
        template = custom_template(background)
    else:
        template = catalog.resolve(template_query)
    if template is None:
        names = ", ".join(f"{t.id} ({t.name})" for t in catalog.suggest(template_query))
        hint = (
            f"Did you mean: {names}?"
            if names
            else "Try @meme list <word>, or paste an image URL as the template."
        )
        return MemePlan(None, None, f"No meme template called '{template_query}'. {hint}")
    lines = [line.strip() for line in lines]
    if not any(lines):
        return MemePlan(None, template, f"Give the captions: {_describe(template)}")
    if len(lines) > template.lines:
        return MemePlan(
            None,
            template,
            f"{template.id} takes {template.lines} captions, you gave {len(lines)}. "
            f"Separate them with |.",
        )
    for line in lines:
        if len(line) > MAX_CAPTION_CHARS:
            return MemePlan(
                None,
                template,
                f"That caption is too long ({len(line)} chars; max {MAX_CAPTION_CHARS}).",
            )
    if options.font and options.font not in FONTS:
        return MemePlan(None, template, f"Fonts: {', '.join(FONTS)}.")
    padded = lines + [""] * (template.lines - len(lines))
    return MemePlan(padded, template, None)


def matches(template: MemeTemplate, word: str) -> bool:
    """Does ``word`` appear in the template's id, name, keywords, or example?"""
    nw = _normalise(word)
    if not nw:
        return True
    return nw in _search_text(template)

"""Build the bundled meme catalog: templates.json + one JPEG per template.

Inputs are the working files of the template harvest (see
docs/plans/2026-10-13-meme-templater.md), none of which are committed:

- a memegen checkout (its templates/<id>/config.yml + default.*: hand-made
  layouts, kept as they are);
- imgflip candidates.json (scraped list, with ``drop`` set by the dedup pass
  to ``memegen:<id>`` where the blank is a memegen template);
- imgflip_meta/<imgflip id>.json, the template's entry from imgflip's own
  meme generator page: ``default_settings`` (the caption boxes imgflip's
  editor opens with, in pixels) and ``altNames``;
- drafts/<imgflip id>.json, the vision model's caption boxes per blank, used
  where imgflip has no default_settings;
- raw/<imgflip id>.<ext>, the blanks;
- an optional rejects file, one template id per line, from the review page.

usage: uv run --with pyyaml python scripts/build_meme_catalog.py \
    --memegen /tmp/memegen-probe/src --work /tmp/memepilot \
    [--rejects rejects.txt] [--out plugins/llm/src/llm/memes]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

MAX_EDGE = 800
JPEG_QUALITY = 85
MIN_BOX_PX = 30
FONTS = {"thick", "thin", "comic"}
STYLES = {"upper", "default", "none", "mock"}
BOX_KEYS = ("anchor_x", "anchor_y", "scale_x", "scale_y")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(name: str, limit: int = 24) -> str:
    """Whole words up to ``limit`` chars: ids go in every picker prompt."""
    out = ""
    for word in _SLUG_RE.sub(" ", name.lower()).split():
        nxt = f"{out}-{word}" if out else word
        if len(nxt) > limit and out:
            break
        out = nxt
    return out[:limit] or "meme"


def norm(name: str) -> str:
    return " ".join(_SLUG_RE.sub(" ", name.lower()).split())


def clean_box(raw: dict, height: int) -> dict | None:
    try:
        box = {k: float(raw[k]) for k in BOX_KEYS}
    except (KeyError, TypeError, ValueError):
        return None
    box["anchor_x"] = min(max(box["anchor_x"], 0.0), 0.95)
    box["anchor_y"] = min(max(box["anchor_y"], 0.0), 0.95)
    box["scale_x"] = min(max(box["scale_x"], 0.05), 1.0 - box["anchor_x"])
    box["scale_y"] = min(max(box["scale_y"], 0.02), 1.0 - box["anchor_y"])
    if box["scale_y"] * height < MIN_BOX_PX:
        return None
    angle = float(raw.get("angle") or 0.0)
    if angle:
        box["angle"] = round(angle, 1)
    color = str(raw.get("color") or "white")
    if color != "white":
        box["color"] = color
    style = str(raw.get("style") or "upper")
    if style != "upper":
        box["style"] = style if style in STYLES else "none"
    font = str(raw.get("font") or "thick")
    if font != "thick":
        box["font"] = font if font in FONTS else "thick"
    align = str(raw.get("align") or "center")
    if align != "center":
        box["align"] = align
    valign = str(raw.get("valign") or "middle")
    if valign in ("top", "bottom"):
        box["valign"] = valign
    for k in BOX_KEYS:
        box[k] = round(box[k], 4)
    return box


_IMGFLIP_FONTS = {"impact": "thick", "comic sans ms": "comic"}


def imgflip_boxes(meta: dict) -> list[dict]:
    """imgflip's ``default_settings`` as memegen-style box dicts.

    imgflip boxes are pixels on a ``w`` x ``h`` blank, rotated clockwise in
    CSS degrees about their centre; ours are fractions, counter-clockwise.
    """
    raw = meta.get("default_settings")
    width, height = meta.get("w"), meta.get("h")
    if not (raw and width and height):
        return []
    try:
        settings = json.loads(raw)
    except ValueError:
        return []
    boxes = []
    for b in settings if isinstance(settings, list) else []:
        if not isinstance(b, dict) or b.get("type", "text") != "text":
            continue
        try:
            x, y, w, h = (float(b[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            continue
        angle = (-float(b.get("rotation") or 0)) % 360
        if angle > 180:
            angle -= 360
        color = str(b.get("font_color") or "#ffffff").lower()
        boxes.append(
            {
                "anchor_x": x / width,
                "anchor_y": y / height,
                "scale_x": w / width,
                "scale_y": h / height,
                "angle": angle,
                "color": {
                    "#ffffff": "white",
                    "#fff": "white",
                    "#000000": "black",
                    "#000": "black",
                }.get(color, color),
                "style": "upper" if b.get("force_caps") else "none",
                "font": _IMGFLIP_FONTS.get(str(b.get("font") or "").lower(), "thin"),
                "align": b.get("text_align") or "center",
                "valign": b.get("vertical_align") or "middle",
            }
        )
    return boxes


def alt_names(meta: dict) -> list[str]:
    return [a.strip() for a in str(meta.get("altNames") or "").split(",") if a.strip()]


def load_meta(work: Path, imgflip_id: str) -> dict:
    path = work / "imgflip_meta" / f"{imgflip_id}.json"
    return json.loads(path.read_text()) if path.exists() else {}


# Flood-fill tolerance for a seed slot: blanks are flat colour plus JPEG
# noise; 40 of 255 per channel holds to the blank without leaking past a
# printed border.
_FILL_THRESH = 40
_FILL_MIN = 0.01


def slot_mask(image: Image.Image, spec: dict) -> tuple[Image.Image, tuple[float, ...]]:
    """An L-mode mask (255 = picture goes here) and its box as fractions.

    ``seed`` flood-fills the blank area from that point, so a curved or
    irregular blank (the costume card) gets an exact outline; the mask is
    grown a pixel to cover the anti-aliased edge. ``rect`` is a plain box.
    """
    from PIL import ImageDraw, ImageFilter

    width, height = image.size
    if "rect" in spec:
        x, y, w, h = spec["rect"]
        mask = Image.new("L", image.size, 0)
        box_px = (int(x * width), int(y * height), int((x + w) * width), int((y + h) * height))
        ImageDraw.Draw(mask).rectangle(box_px, fill=255)
    else:
        sx, sy = spec["seed"]
        marked = image.convert("RGB").copy()
        sentinel = (255, 0, 254)
        ImageDraw.floodfill(
            marked, (int(sx * width), int(sy * height)), sentinel, thresh=_FILL_THRESH
        )
        mask = Image.new("L", image.size, 0)
        mask.putdata([255 if p == sentinel else 0 for p in marked.getdata()])
        mask = mask.filter(ImageFilter.MaxFilter(3))
    bbox = mask.getbbox()
    if bbox is None or (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < _FILL_MIN * width * height:
        raise ValueError(f"slot {spec} covers almost nothing")
    left, top, right, bottom = bbox
    return mask, (left / width, top / height, (right - left) / width, (bottom - top) / height)


def mask_corners(mask: Image.Image) -> list[list[float]]:
    """Top-left, top-right, bottom-right, bottom-left of a tilted quad mask,
    as fractions: the extreme top, right, bottom and left pixels, which is
    which corner for a sheet turned less than 45 degrees clockwise."""
    width, height = mask.size
    data = mask.load()
    assert data is not None
    points = [(x, y) for y in range(height) for x in range(width) if data[x, y]]
    top = min(points, key=lambda p: (p[1], p[0]))
    right = max(points, key=lambda p: (p[0], -p[1]))
    bottom = max(points, key=lambda p: (p[1], p[0]))
    left = min(points, key=lambda p: (p[0], -p[1]))
    return [[round(x / width, 4), round(y / height, 4)] for x, y in (top, right, bottom, left)]


def save_jpeg(src: Path, dest: Path) -> tuple[int, int]:
    image = Image.open(src)
    image.seek(0)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        flat = Image.new("RGB", image.size, "white")
        flat.paste(image, mask=image.split()[-1])
        image = flat
    else:
        image = image.convert("RGB")
    image.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
    image.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return image.size


def words(items) -> list[str]:
    return [str(x).strip() for x in items or [] if isinstance(x, str) and str(x).strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memegen", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--drafts", default="drafts/sonnet")
    ap.add_argument("--rejects", type=Path)
    ap.add_argument(
        "--validated", type=Path, default=Path(__file__).with_name("meme_validated.json")
    )
    ap.add_argument("--slots", type=Path, default=Path(__file__).with_name("meme_slots.json"))
    ap.add_argument("--out", type=Path, default=Path("plugins/llm/src/llm/memes"))
    args = ap.parse_args()

    rejects = set()
    if args.rejects and args.rejects.exists():
        rejects = {ln.strip() for ln in args.rejects.read_text().split() if ln.strip()}
    candidates = json.loads((args.work / "candidates.json").read_text())
    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for old in images_dir.glob("*.jpg"):
        old.unlink()
    masks_dir = args.out / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for old in masks_dir.glob("*.png"):
        old.unlink()

    # imgflip's name for a memegen template is what people type ("Two
    # Buttons", not "Daily Struggle"), and its rank is the popularity order.
    mg_alias: dict[str, list[str]] = {}
    mg_rank: dict[str, int] = {}
    for c in candidates:
        drop = c.get("drop") or ""
        if drop.startswith("memegen:"):
            mid = drop.split(":", 1)[1]
            mg_alias.setdefault(mid, []).extend(
                [c["name"], *alt_names(load_meta(args.work, c["id"]))]
            )
            mg_rank[mid] = min(mg_rank.get(mid, 10**6), c["rank"])

    entries: list[dict] = []
    taken_ids: set[str] = set()
    taken_names: set[str] = set()
    stats = {
        "memegen": 0,
        "imgflip_boxes": 0,
        "drafted_boxes": 0,
        "rejected": 0,
        "unusable": 0,
        "no_boxes": 0,
        "name_dup": 0,
        "no_layout": 0,
    }

    for tdir in sorted((args.memegen / "templates").iterdir()):
        mid = tdir.name
        if mid.startswith("_") or mid in rejects:
            continue
        cfg_path = tdir / "config.yml"
        image = next(iter(sorted(tdir.glob("default.*"))), None)
        if not cfg_path.exists() or image is None:
            continue
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        w, h = save_jpeg(image, images_dir / f"{mid}.jpg")
        boxes = [b for b in (clean_box(t, h) for t in cfg.get("text") or []) if b]
        if not boxes:
            (images_dir / f"{mid}.jpg").unlink()
            stats["no_boxes"] += 1
            continue
        aliases = mg_alias.get(mid, [])
        entries.append(
            {
                "id": mid,
                "name": str(cfg.get("name") or mid),
                "keywords": [*aliases, *words(cfg.get("keywords"))],
                "example": [str(x) for x in (cfg.get("example") or [])][: len(boxes)],
                "rank": mg_rank.get(mid, 5000),
                "source": "memegen",
                "layout": "memegen",
                "boxes": boxes,
            }
        )
        taken_ids.add(mid)
        taken_names.update(norm(n) for n in [entries[-1]["name"], *aliases[:1]])
        stats["memegen"] += 1

    for c in candidates:
        if c.get("drop"):
            continue
        draft_path = args.work / args.drafts / f"{c['id']}.json"
        draft = json.loads(draft_path.read_text()) if draft_path.exists() else {}
        meta = load_meta(args.work, c["id"])
        stolen = imgflip_boxes(meta)
        if not stolen and not draft:
            stats["no_layout"] += 1
            continue
        if draft.get("usable") is False:
            stats["unusable"] += 1
            continue
        names = [c["name"], str(draft.get("name") or "")]
        if any(norm(n) in taken_names for n in names if n):
            stats["name_dup"] += 1
            continue
        tid = slug(c["name"])
        base, n = tid, 2
        while tid in taken_ids:
            tid, n = f"{base}-{n}", n + 1
        if tid in rejects or c["id"] in rejects:
            stats["rejected"] += 1
            continue
        raw = next((args.work / "raw").glob(f"{c['id']}.*"))
        w, h = save_jpeg(raw, images_dir / f"{tid}.jpg")
        raw_boxes = stolen or draft.get("text") or []
        boxes = [b for b in (clean_box(t, h) for t in raw_boxes) if b]
        if not boxes:
            (images_dir / f"{tid}.jpg").unlink()
            stats["no_boxes"] += 1
            continue
        stats["imgflip_boxes" if stolen else "drafted_boxes"] += 1
        # The draft's example captions follow the draft's boxes; with
        # imgflip's boxes they still read in order, and a count mismatch
        # just leaves boxes blank in the review render.
        example = [str(x) for x in draft.get("example") or []][: len(boxes)]
        keywords = [*alt_names(meta), *words(draft.get("aliases")), *words(draft.get("keywords"))]
        if draft.get("name") and norm(draft["name"]) != norm(c["name"]):
            keywords.insert(0, str(draft["name"]))
        entries.append(
            {
                "id": tid,
                "name": c["name"],
                "keywords": keywords,
                "example": example,
                "rank": c["rank"],
                "source": f"imgflip:{c['id']}",
                # imgflip: the editor's preset boxes; drafted: a vision model's
                # guess, unreviewed unless "validated" says otherwise.
                "layout": "imgflip" if stolen else "drafted",
                "boxes": boxes,
            }
        )
        taken_ids.add(tid)
        taken_names.update(norm(x) for x in names if x)

    # Layouts already checked by eye: marked so nobody re-renders them.
    checked = json.loads(args.validated.read_text()) if args.validated.exists() else {}
    ok, issues = set(checked.get("ok", [])), checked.get("issues", {})
    for e in entries:
        if e["id"] in ok:
            e["validated"] = True
        elif e["id"] in issues:
            e["issue"] = issues[e["id"]]
    stats["validated"] = sum(1 for e in entries if e.get("validated"))

    # Picture slots: the mask says exactly where a generated picture goes.
    slot_spec = json.loads(args.slots.read_text()) if args.slots.exists() else {}
    by_id = {e["id"]: e for e in entries}
    for tid, specs in slot_spec.items():
        if tid.startswith("_"):
            continue
        if tid not in by_id:
            raise SystemExit(f"meme_slots.json names {tid!r}, which is not in the catalog")
        image = Image.open(images_dir / f"{tid}.jpg")
        slots = []
        for i, spec in enumerate(specs):
            mask, box = slot_mask(image, spec)
            name = f"{tid}-{i}.png"
            mask.convert("1").save(masks_dir / name, optimize=True)
            slot: dict[str, Any] = {"mask": name, "box": [round(b, 4) for b in box]}
            if spec.get("corners"):
                slot["corners"] = mask_corners(mask)
            slots.append(slot)
        by_id[tid]["slots"] = slots
    stats["slotted"] = sum(1 for e in entries if e.get("slots"))
    stats["issues"] = sum(1 for e in entries if e.get("issue"))

    entries.sort(key=lambda e: (e["rank"], e["id"]))
    (args.out / "templates.json").write_text(
        json.dumps(entries, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    size = sum(p.stat().st_size for p in images_dir.glob("*.jpg"))
    print(json.dumps(stats), f"total={len(entries)}", f"images={size / 1e6:.1f}MB")


if __name__ == "__main__":
    main()

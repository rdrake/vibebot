"""Render every catalog template with its example captions into a review page.

The page shows each template as the bot would draw it, box outlines on,
with a reject checkbox; "Copy rejected ids" puts the ids on the clipboard
for build_meme_catalog.py --rejects.

usage: uv run python scripts/meme_review_page.py OUT_DIR [--flags DRAFTS_DIR]
       [--catalog plugins/llm/src/llm/memes/templates.json]

--flags points at the vision drafts so templates whose draft named a
watermark are marked and listed first.
"""

from __future__ import annotations

import argparse
import html
import io
import json
from pathlib import Path

from llm import meme, meme_render
from PIL import Image, ImageDraw

TILE = 420


def tile(template: meme.MemeTemplate) -> bytes:
    assert template.image is not None
    image = meme_render.load_image(template.image)
    lines = list(template.example) or [template.name] * template.lines
    png = meme_render.render(image, template.boxes, lines)
    out = Image.open(io.BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(out)
    width, height = out.size
    for box in template.boxes:
        x, y = width * box.anchor_x, height * box.anchor_y
        draw.rectangle(
            (x, y, x + width * box.scale_x, y + height * box.scale_y), outline="lime", width=2
        )
    out.thumbnail((TILE, TILE))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--catalog", type=Path, default=meme.CATALOG_PATH)
    ap.add_argument("--flags", type=Path)
    args = ap.parse_args()

    catalog = meme.bundled_catalog(args.catalog)
    assert catalog is not None, f"no catalog at {args.catalog}"
    entries = {e["id"]: e for e in json.loads(args.catalog.read_text())}
    tiles = args.out / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)

    flags: dict[str, str] = {}
    if args.flags:
        for t in catalog.templates:
            source = entries[t.id].get("source", "")
            if source.startswith("imgflip:"):
                draft = args.flags / f"{source.split(':', 1)[1]}.json"
                if draft.exists():
                    mark = json.loads(draft.read_text()).get("watermark")
                    if mark:
                        flags[t.id] = str(mark)

    ordered = sorted(catalog.templates, key=lambda t: (t.id not in flags, t.rank))
    cards = []
    for t in ordered:
        (tiles / f"{t.id}.jpg").write_bytes(tile(t))
        source = entries[t.id].get("source", "")
        flag = (
            f'<div class="flag">watermark? {html.escape(flags[t.id])}</div>'
            if t.id in flags
            else ""
        )
        cards.append(
            f'<label class="card"><img loading="lazy" src="tiles/{t.id}.jpg">'
            f'<div><input type="checkbox" value="{html.escape(t.id)}"> '
            f"<b>{html.escape(t.id)}</b> #{t.rank} {html.escape(source)}</div>"
            f"<div>{html.escape(t.name)}</div>{flag}</label>"
        )
    page = f"""<!doctype html><meta charset="utf-8"><title>Meme review</title>
<style>
body{{font:13px system-ui;margin:0;background:#222;color:#ddd}}
header{{position:sticky;top:0;background:#111;padding:8px;z-index:1}}
main{{display:flex;flex-wrap:wrap;gap:8px;padding:8px}}
.card{{width:{TILE}px;background:#333;padding:4px;cursor:pointer}}
.card img{{max-width:100%;display:block}}
.card:has(input:checked){{outline:4px solid #e33;opacity:.6}}
.flag{{color:#fc0}}
textarea{{width:60%;height:2.4em;vertical-align:middle}}
</style>
<header>{len(cards)} templates · <span id="n">0</span> rejected
<button onclick="copyIds()">Copy rejected ids</button>
<textarea id="ids" readonly></textarea></header>
<main>{"".join(cards)}</main>
<script>
const KEY='meme-review-rejects';
const saved=new Set(JSON.parse(localStorage.getItem(KEY)||'[]'));
const boxes=[...document.querySelectorAll('input[type=checkbox]')];
function sync(){{
  const ids=boxes.filter(b=>b.checked).map(b=>b.value);
  localStorage.setItem(KEY,JSON.stringify(ids));
  document.getElementById('n').textContent=ids.length;
  document.getElementById('ids').value=ids.join(' ');
}}
boxes.forEach(b=>{{b.checked=saved.has(b.value);b.addEventListener('change',sync)}});
sync();
function copyIds(){{const t=document.getElementById('ids');t.select();navigator.clipboard.writeText(t.value)}}
</script>
"""
    (args.out / "index.html").write_text(page)
    print(f"{len(cards)} templates, {len(flags)} flagged -> {args.out / 'index.html'}")


if __name__ == "__main__":
    main()

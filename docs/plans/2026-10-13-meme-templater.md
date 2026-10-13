# Own meme templater + expanded template catalog

Goal: replace memegen.link with an in-plugin renderer and grow the catalog from
memegen's ~200 templates to as many clean (watermark-free) templates as review
time allows.

## Decisions (2026-10-13)

- **Render in the plugin with Pillow (Option B).** Self-hosted memegen measured
  142 MiB idle / 165 MiB after a GIF render (gunicorn + 1 uvicorn worker, local
  amd64 container). Prod has 119 MB available, no swap → does not fit.
  Pillow is already a dependency of `plugins/llm`.
- **Vendor memegen's text fitting (MIT)** — `app/utils/images.py` `wrap`,
  `split_2/3`, `get_font`, `get_text_offset`, `get_text_size`,
  `get_stroke_width`, plus `Text.get_anchor/get_size/get_stroke/stylize`
  (~200 lines). Keep attribution.
- **Template format = memegen `config.yml`** (`text:` boxes with
  `anchor_x/anchor_y/scale_x/scale_y/angle/color/font/align`, `example`,
  `keywords`). The 211 existing memegen layouts carry over unchanged.
- **Fonts:** ship OFL fonts only (TitilliumWeb-Black = memegen "thick" default,
  Kalam, NotoSans). memegen's `Impact.ttf` is Microsoft's — do NOT ship it; use
  Anton (OFL) if an Impact look is wanted.
- **Sources:** imgflip (`get_memes` top 100, then `/memetemplates` pages) for
  blank images; Know Your Meme for names/aliases/keywords; memegen's repo for
  existing layouts.
- **Watermarks:** imgflip blanks checked 10/10 clean of imgflip marks (only the
  original artists' signatures, e.g. Two Buttons' `JAKE-CLARK.TUMBLR`). OCR
  filter only needed for non-imgflip sources.
- **Box coordinates:** drafted by a vision model, rendered with example
  captions, approved by rdrake on an HTML contact sheet. Pilot measures yield
  before scaling.
- **Aspect guard:** swapping a cleaner/larger image onto an existing layout
  only if aspect ratio matches within ±2%; otherwise re-draft boxes.
- **Memory:** open each template image per render; never preload the catalog.

## Open questions

- Where template images live: baked into the ghcr image (simplest with
  auto-deploy; ~500 templates ≈ 50 MB) vs `/config` or boxlabs.
- Target count — decided after the pilot yield is known.

## Breaking changes (no backcompat)

- `memeApiBase` removed; its persisted value in prod `bot.conf` becomes dead.
- `--gif` (animated variants) and `--style` (alternate images) go away unless
  the vendored renderer keeps per-template alternates.
- `meme.py` escape table, `canonical_url`, `_NoRedirect`, `fetch_templates`
  become deletable.
- `@meme` / `make_meme` docs that mention memegen need updating.

## Steps

1. [x] Measure memegen RAM → Option B.
2. [x] Check imgflip blanks for watermarks → clean.
3. [x] Pilot: 20 imgflip templates → vision-model boxes → Pillow render →
       contact sheet. Result (Sonnet via `claude -p`, one pass, 112 s for 20
       at 5-way parallel): 17/20 usable as drafted, 3 need a small fix —
       Bernie (bottom box overlaps the baked-in caption), 0 Days Without
       (extra second box), Absolute Cinema (top box too thin, text tiny).
       db and gru match memegen's hand layouts. Pilot scripts and sheets in
       /tmp/memepilot (render.py, draft.py, draft_prompt.md, sheet.py).
4. [x] rdrake picked 1000. Harvest: imgflip /memetemplates pages 1-45 →
       1,182 after phash dedup (vs memegen and each other) and min-side
       250 px. Draft prompt v2 adds usable/watermark flags, no-overlap with
       baked text, min box height 45 px. Self-review pass tested and DROPPED
       (changed 1/6, fixed 0/3 failures). Full draft run started 18:07,
       ~18/min. First 40: ~36 good. Watermark flags go to review, not
       auto-reject (Bernie/WWE broadcaster logos flagged too).

### Pivot 2026-10-13 18:30 — steal imgflip's own boxes

rdrake asked "can you steal the coordinates from an API?". Yes: imgflip's
`/memegenerator/<id>/x` page embeds `memes=` JSON (list or `{"0":..,"U":..}`
object; "U" is the requested template) whose `default_settings` holds the
editor's preset boxes — pixel x/y/w/h on the `w`x`h` blank, CSS rotation
(clockwise, about the centre), font (impact/arial), font_color, force_caps,
text_align, vertical_align — plus `altNames`. Sampled: most templates have
it; some (Roll Safe) don't. Scraper: /tmp/memepilot/scripts/scrape_layouts.py
→ imgflip_meta/<id>.json. Build priority: memegen layout (its 209) → imgflip
default_settings → vision draft. Renderer gained valign, right align,
centre-rotation, full white outline on black text. Rotation sign verified
visually (Epic Handshake, Left Exit car).

### Plugin refactor touch list (step 5)

- `meme_render.py` (new): vendored renderer + `style: default` + pad_top
  for `--top` + custom two-band over supplied bytes. Attribution header.
- `meme.py`: MemeTemplate gains boxes + image path, loses styles/animated.
  Delete URL/escape/fetch/canonical/CachedCatalog code. `load_catalog(dir)`
  reads one `templates.json` (mtime-checked). Font check only.
- `service.py`: split `_download_and_save_image` → `_download_image_bytes`
  (guards stay) + save. Needed for pasted-URL and URL-alias templates.
- `plugin.py`: render → `_save_image_bytes`; drop `--gif`/`--style`;
  usage row model `memegen` → `local`.
- `config.py`: remove `memeApiBase`.
- Fonts under `llm/fonts/` with OFL; verify package data includes .ttf/.json.
  Map memegen `tiny`/Tahoma configs to notosans.
- Picker brief: top ~180 by popularity + ≤60 lexical hits, cap ~250 rows.
- Import: drafts + memegen 211 → `templates.json` + `images/<id>.jpg`
  (≤800 px, q85). Fold imgflip names onto memegen dups as keywords. Second
  name-dedup pass (phash misses re-crops, e.g. Drake 1:1 vs 0.64).
- Review: local HTML page with checkboxes → "copy rejected ids".
- Storage: recommend committing images (~50-60 MB; .git is 21 MB now;
  prod 9.5 GB free) — ASK rdrake.
5. [ ] Vendor renderer into `plugins/llm/src/llm/meme_render.py`; swap
       `meme.py` from URL-building to local render; tests.
6. [ ] Harvest + dedup (phash vs memegen's 211) + box drafting at scale.
7. [ ] Review sheets; commit approved templates; picker shortlist so the
       catalog brief stays small.
8. [ ] Deploy; list prod config changes.

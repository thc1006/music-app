"""
Phase D1 — Render Bach chorales as clean OMR-friendly PNG images.

Strategy:
  1. Load chorale via music21
  2. Strip lyrics and tempo/title metadata
  3. Export to LilyPond source (.ly)
  4. Wrap with custom \\paper {} block (compact, no titles, large staff)
  5. Run lilypond with -dresolution=300 → high-DPI PNG
  6. Verify Phase 9 detects ≥80% of expected noteheads

Output: training/datasets/chorale_gt/{bwvNNN}.png + ground truth metadata
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/thc1006/dev/music-app")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

OUT_DIR = PROJECT_ROOT / "training/datasets/chorale_gt"
WORK_DIR = OUT_DIR / "_lilypond_work"

# Compact paper settings — no titles, minimal margins, larger staff
PAPER_BLOCK = r"""
\paper {
  paper-width = 11\in
  paper-height = 8.5\in
  top-margin = 0.4\in
  bottom-margin = 0.4\in
  left-margin = 0.5\in
  right-margin = 0.5\in
  print-page-number = ##f
  print-first-page-number = ##f
  oddHeaderMarkup = ##f
  evenHeaderMarkup = ##f
  oddFooterMarkup = ##f
  evenFooterMarkup = ##f
  scoreTitleMarkup = ##f
  bookTitleMarkup = ##f
  ragged-bottom = ##t
  ragged-last-bottom = ##t
}

#(set-global-staff-size 22)
"""


def strip_lyrics_and_titles(score):
    """Remove lyrics and metadata that would clutter the rendered image."""
    from music21 import note, metadata

    # Strip lyrics from every note
    for n in score.recurse().getElementsByClass(note.Note):
        n.lyrics = []
    for c in score.recurse().getElementsByClass(note.Rest):
        c.lyrics = []

    # Clear metadata (title, composer, etc)
    score.metadata = metadata.Metadata()
    return score


def patch_ly_source(ly_text: str) -> str:
    r"""Insert our compact \paper{} block into a music21-generated .ly file."""
    # Find the \version line and insert paper block right after it
    lines = ly_text.splitlines()
    out = []
    inserted = False
    for line in lines:
        out.append(line)
        if not inserted and line.strip().startswith(r"\version"):
            out.append(PAPER_BLOCK)
            inserted = True
    if not inserted:
        out.insert(0, PAPER_BLOCK)
    return "\n".join(out)


def render_chorale(bwv: str) -> Path | None:
    """Render one chorale to a single PNG. Returns path or None on failure."""
    from music21 import corpus

    print(f"\n--- Rendering bach/bwv{bwv} ---")
    try:
        score = corpus.parse(f"bach/bwv{bwv}")
    except Exception as e:
        print(f"  parse failed: {e}")
        return None

    score = strip_lyrics_and_titles(score)

    # Step 1: write LilyPond source via music21
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ly_base = WORK_DIR / f"bwv{bwv}"
    try:
        ly_path_str = score.write(fmt="lilypond", fp=str(ly_base))
    except Exception as e:
        print(f"  music21 lilypond export failed: {e}")
        return None
    ly_path = Path(ly_path_str)
    if not ly_path.exists():
        # music21 sometimes adds .ly extension
        ly_path = Path(str(ly_path_str) + ".ly")
    if not ly_path.exists():
        print(f"  cannot find .ly output (tried {ly_path_str})")
        return None

    # Step 2: patch the .ly source with our paper block
    src = ly_path.read_text()
    patched = patch_ly_source(src)
    ly_path.write_text(patched)

    # Step 3: run lilypond with high resolution PNG output
    output_base = WORK_DIR / f"bwv{bwv}_render"
    try:
        result = subprocess.run(
            [
                "lilypond",
                "--png",
                "-dresolution=300",
                f"-o{output_base}",
                str(ly_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("  lilypond timed out")
        return None
    if result.returncode != 0:
        print(f"  lilypond failed: {result.stderr[-300:]}")
        return None

    # Find the produced PNG (lilypond may add -page1 etc.)
    candidates = sorted(WORK_DIR.glob(f"bwv{bwv}_render*.png"))
    if not candidates:
        print("  no PNG output")
        return None

    # Pick page 1 if multipage, copy to OUT_DIR
    page1 = candidates[0]
    final = OUT_DIR / f"bwv{bwv}.png"
    shutil.copy(page1, final)
    print(f"  -> {final} ({len(candidates)} page(s) total)")
    return final


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5 candidate chorales (well-known, varied keys, varied lengths)
    candidates = [
        "269",  # Aus meines Herzens Grunde
        "267",  # Sei Lob und Ehr
        "270",  # Befiehl du deine Wege
        "354",  # Es ist genug
        "377",  # Christus, der ist mein Leben
    ]
    rendered = []
    for bwv in candidates:
        p = render_chorale(bwv)
        if p:
            rendered.append(p)
    print()
    print(f"Successfully rendered {len(rendered)}/{len(candidates)}:")
    for p in rendered:
        print(f"  {p}")


if __name__ == "__main__":
    main()

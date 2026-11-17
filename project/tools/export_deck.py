"""Renders an HTML deck to 1920x1080 images and an image-only PPTX backup (no animation, last step of each slide).

Usage:
  python tools/export_deck.py DECK.html               # all slides -> <deck>-frames/sN.png and <deck>.pptx
  python tools/export_deck.py DECK.html 3 5           # only slides 3 and 5 (for checking), no PPTX
  python tools/export_deck.py DECK.html --out DIR     # write frames and PPTX to DIR (default: next to the deck)
Console errors from the page are printed.

Requires: playwright (with Chrome) and python-pptx.
"""
import pathlib
import sys

from playwright.sync_api import sync_playwright


def shots(page_url, out, which):
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.on("console", lambda m: m.type in ("error", "warning") and errors.append(m.text))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(page_url + "?export#slide-1")
        page.evaluate("document.fonts.ready")
        n = page.evaluate("document.querySelectorAll('.slide').length")
        for i in which or range(1, n + 1):
            page.goto(f"{page_url}?export#slide-{i}.end")
            page.evaluate("document.fonts.ready")
            page.wait_for_timeout(500)
            page.screenshot(path=str(out / f"s{i}.png"))
        browser.close()
    return n, errors


def pptx(n, frames, target):
    from pptx import Presentation
    from pptx.util import Emu
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
    for i in range(1, n + 1):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        s.shapes.add_picture(str(frames / f"s{i}.png"), 0, 0, prs.slide_width, prs.slide_height)
    prs.save(target)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = pathlib.Path(args[i + 1]).resolve()
        del args[i:i + 2]
    deck = pathlib.Path(args.pop(0)).resolve()
    out = out or deck.parent
    frames = out / f"{deck.stem}-frames"
    which = [int(a) for a in args]
    n, errors = shots(deck.as_uri(), frames, which)
    print("console:", errors or "clean")
    if not which:
        target = out / f"{deck.stem}.pptx"
        pptx(n, frames, target)
        print(f"{n} slides -> {target}")

"""Markdown -> styled HTML -> PDF with headless Chrome. Intermediate files go to a temporary folder.

Usage:
  python tools/md_to_pdf.py DOC.md [OUT.pdf] [--title T] [--kicker K] [--authors A] [--lang L]

The first line of DOC.md (its "# ..." heading) is replaced by the header block. Without OUT.pdf the PDF is
written next to DOC.md. Chrome is found automatically; set the CHROME environment variable to override.

Requires: markdown-it-py, Google Chrome (or Chromium).
"""
import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

from markdown_it import MarkdownIt

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def find_chrome():
    if os.environ.get("CHROME"):
        return os.environ["CHROME"]
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        if shutil.which(name):
            return shutil.which(name)
    for path in CHROME_PATHS:
        if pathlib.Path(path).exists():
            return path
    sys.exit("Chrome not found; set the CHROME environment variable to its executable.")


CSS = """
@page { size: A4; margin: 13mm 14mm 13mm; }
* { box-sizing: border-box; }
body { margin: 0; font: 9.4pt/1.5 Inter, "Segoe UI", Arial, sans-serif; color: #1B2228; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
h1, h2, h3, .kick { font-family: "Space Grotesk", "Segoe UI", sans-serif; }
.top { border-bottom: 2px solid #1B2228; padding-bottom: 7px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: flex-end; }
.kick { font-size: 8pt; letter-spacing: .16em; color: #C98A1E; font-weight: 600; text-transform: uppercase; }
h1 { font-size: 20pt; margin: 3px 0 0; line-height: 1.1; }
.who { font-size: 8.6pt; color: #5E6A72; text-align: right; }
h2 { font-size: 12.5pt; margin: 16px 0 6px; padding: 5px 10px; background: #F6F3EC; border-left: 4px solid #C98A1E; border-radius: 0 6px 6px 0; break-after: avoid; }
h3 { font-size: 10.8pt; margin: 12px 0 4px; color: #0E1216; border-bottom: 1px solid #E3E0D9; padding-bottom: 2px; break-after: avoid; }
p { margin: 0 0 5px; }
ul { margin: 0 0 6px; padding-left: 18px; }
ol { margin: 0 0 6px; padding-left: 24px; }
li { margin: 1px 0; }
li > ul, li > ol { margin: 2px 0; }
p.q { margin: 9px 0 2px; font-size: 9.8pt; color: #0E1216; break-after: avoid; }
p.q strong { font-family: "Space Grotesk", "Segoe UI", sans-serif; font-weight: 600; }
p.q + ul, p.q + ol, p.q + blockquote { break-before: avoid; }
blockquote { margin: 4px 0 6px; padding: 6px 10px; background: #EEF6F3; border-left: 3px solid #3E9C86; border-radius: 0 6px 6px 0; break-inside: avoid; }
blockquote p { margin: 0; }
blockquote strong { color: #2F7F6C; }
table { border-collapse: collapse; width: 100%; margin: 4px 0 8px; font-size: 8.5pt; }
th { text-align: left; background: #1B2228; color: #fff; font-weight: 600; padding: 4px 6px; vertical-align: bottom; }
td { border-bottom: 1px solid #E3E0D9; padding: 3.5px 6px; vertical-align: top; }
tr { break-inside: avoid; }
td:first-child { font-weight: 600; }
hr { border: 0; height: 0; margin: 4px 0; }
sup, sub { font-size: .72em; line-height: 0; }
code { font-family: Consolas, monospace; font-size: .92em; background: #F1EFE9; padding: 0 3px; border-radius: 3px; }
a { color: inherit; text-decoration: none; }
"""


def build(src, out, title, kicker, authors, lang):
    md = src.read_text(encoding="utf-8")
    md = re.sub(r"^\*\*PDF version:\*\*.*\n", "", md, flags=re.M)   # a link to itself is pointless inside the PDF
    first, _, rest = md.partition("\n")
    title = title or first.lstrip("# ").strip()
    body = MarkdownIt("commonmark").enable("table").render(rest)
    # paragraphs made only of bold text are questions or sub-headings
    body = re.sub(r"<p><strong>([^<]*)</strong></p>", r'<p class="q"><strong>\1</strong></p>', body)
    body = re.sub(r"(\w)\^\(([^)]*)\)", r"\1<sup>\2</sup>", body)
    body = re.sub(r"(\d)\^(\d+)", r"\1<sup>\2</sup>", body)
    html = f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<div class="top"><div><div class="kick">{kicker}</div><h1>{title}</h1></div><div class="who">{authors}</div></div>
{body}
</body></html>"""
    with tempfile.TemporaryDirectory() as t:
        page = pathlib.Path(t) / "page.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run([find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                        f"--user-data-dir={pathlib.Path(t) / 'chrome'}", f"--print-to-pdf={out}", page.as_uri()],
                       check=True, capture_output=True)
    print("ok", out, out.stat().st_size)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Markdown -> PDF via headless Chrome")
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path, nargs="?")
    ap.add_argument("--title", default="")
    ap.add_argument("--kicker", default="Introduction to AI · Team Eight Bits")
    ap.add_argument("--authors", default="Eren Terakye · Salih Eren")
    ap.add_argument("--lang", default="en")
    a = ap.parse_args()
    build(a.src.resolve(), (a.out or a.src.with_suffix(".pdf")).resolve(), a.title, a.kicker, a.authors, a.lang)

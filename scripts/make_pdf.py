"""Convert report/report.md to report/report.pdf.

Markdown is rendered to HTML with mistune (tables enabled), styled for A4
printing, and printed to PDF by a Chromium-based browser in headless mode
(Microsoft Edge by default, Google Chrome as fallback). Image paths in the
Markdown (e.g. ../results/figures/*.png) are resolved relative to the
Markdown file. The intermediate HTML is written to a temporary directory and
deleted afterwards.

Usage:
    python scripts/make_pdf.py [--input report/report.md] [--output report/report.pdf]
                               [--browser PATH]
"""

import argparse
import html as htmllib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote

import mistune

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.42; color: #111; }
h1 { font-size: 20pt; margin: 0 0 4pt 0; }
h2 { font-size: 14pt; margin: 18pt 0 6pt 0; border-bottom: 1px solid #999;
     padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt 0; break-after: avoid; }
p, li { text-align: justify; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt;
       background: #f2f2f2; padding: 0 2px; }
table { border-collapse: collapse; margin: 6pt 0 10pt 0; font-size: 9pt;
        break-inside: avoid; }
th, td { border: 1px solid #888; padding: 2pt 5pt; vertical-align: top; }
th { background: #e8e8e8; }
img { display: block; max-width: 100%; max-height: 105mm; margin: 6pt auto 2pt auto;
      break-inside: avoid; }
p:has(> img) { break-after: avoid; text-align: center; }
blockquote { margin: 6pt 0 6pt 12pt; padding-left: 8pt; border-left: 3px solid #999;
             color: #333; }
hr { border: none; border-top: 1px solid #bbb; margin: 10pt 0; }
"""


def find_browser(explicit):
    if explicit:
        if not explicit.is_file():
            sys.exit(f"error: browser not found: {explicit}")
        return explicit
    for path in BROWSERS:
        if path.is_file():
            return path
    sys.exit("error: no Edge/Chrome found; pass --browser PATH")


def render_html(md_path):
    """Markdown -> standalone HTML with absolute file:// image URLs."""
    markdown = mistune.create_markdown(plugins=["table", "strikethrough"])
    body = markdown(md_path.read_text(encoding="utf-8"))
    missing = []

    def absolute(match):
        src = match.group(2)
        if re.match(r"^[a-z]+:", src):  # already a URL
            return match.group(0)
        target = (md_path.parent / unquote(src)).resolve()  # mistune %-encodes spaces
        if not target.is_file():
            missing.append(src)
        return f'{match.group(1)}{target.as_uri()}{match.group(3)}'

    body = re.sub(r'(<img src=")([^"]+)(")', absolute, body)
    if missing:
        sys.exit(f"error: images not found: {missing}")
    title = htmllib.escape(md_path.stem)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>{title}</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path, default=ROOT / "report" / "report.md")
    ap.add_argument("--output", type=Path, default=ROOT / "report" / "report.pdf")
    ap.add_argument("--browser", type=Path, help="Edge/Chrome executable")
    args = ap.parse_args()

    md_path = args.input.resolve()
    out = args.output.resolve()
    browser = find_browser(args.browser)
    out.parent.mkdir(parents=True, exist_ok=True)  # the browser does not create it
    tmp = Path(tempfile.mkdtemp(prefix="report_pdf_"))
    try:
        html = tmp / "report.html"
        html.write_text(render_html(md_path), encoding="utf-8")
        # A private profile avoids clashing with a browser window already open.
        cmd = [str(browser), "--headless", "--disable-gpu", "--no-pdf-header-footer",
               f"--user-data-dir={tmp / 'profile'}", f"--print-to-pdf={out}",
               html.as_uri()]
        before = out.stat().st_mtime if out.exists() else None
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if (proc.returncode != 0 or not out.is_file()
                or (before is not None and out.stat().st_mtime == before)):
            sys.exit(f"error: PDF not produced (rc {proc.returncode}): {proc.stderr.strip()}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    data = out.read_bytes()
    if not data.startswith(b"%PDF"):
        sys.exit(f"error: {out} is not a PDF")
    pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
    print(f"{out} : {len(data) / 1e6:.2f} MB, {pages} pages ({browser.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

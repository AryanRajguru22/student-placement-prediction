"""Render report/Student_Placement_Predictor_Complete_Project_Guide.md to HTML and PDF (presentation only; no analysis, no data files read).

Usage (from the project root):  python report/guide_assets/build_guide.py
Needs: markdown, pymupdf (fitz), and Microsoft Edge or Chrome for headless printing.
Two passes: the first PDF is used to find the page of each chapter, which is then written into the table of contents.
"""
import base64
import re
import subprocess
import tempfile
from pathlib import Path

import fitz
import markdown

HERE = Path(__file__).resolve().parent.parent          # report/
SRC = HERE / "Student_Placement_Predictor_Complete_Project_Guide.md"
HTML = HERE / "Student_Placement_Predictor_Complete_Project_Guide.html"
PDF = HERE / "Student_Placement_Predictor_Complete_Project_Guide.pdf"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"]

CSS = """
:root{--navy:#12355b;--line:#c9d3df;--soft:#eef2f7}
@page{size:A4;margin:16mm 15mm 18mm 15mm;@bottom-center{content:counter(page);font:9pt Helvetica,Arial,sans-serif;color:#5b6670}}
@page :first{@bottom-center{content:""}}
*{box-sizing:border-box}
body{font-family:Georgia,'Times New Roman',serif;font-size:10pt;line-height:1.4;color:#1b1b1b;margin:0}
h1,h2,h3{font-family:Helvetica,Arial,sans-serif;color:var(--navy)}
h1{font-size:19pt;border-bottom:2px solid var(--navy);padding-bottom:4px;margin:0 0 10px;page-break-before:always}
h1.cont{page-break-before:avoid;margin-top:18px}
h2{font-size:12.5pt;margin:16px 0 5px;border-bottom:1px solid var(--line);padding-bottom:2px;page-break-after:avoid}
p{margin:5px 0}
table{border-collapse:collapse;margin:8px 0;font-size:8.7pt;width:100%;page-break-inside:auto}
tr{page-break-inside:avoid}
th,td{border:1px solid var(--line);padding:3px 6px;vertical-align:top;text-align:left}
th{background:var(--soft);font-family:Helvetica,Arial,sans-serif}
code{background:#f1f1f1;padding:0 2px;font-size:.88em;word-break:break-word}
ul,ol{margin:4px 0 4px 18px;padding:0}li{margin:2px 0}
img{max-width:100%;display:block;margin:6px auto;max-height:9cm;width:auto}
.eq{font-family:'Cambria Math',Cambria,Georgia,serif;font-size:11.5pt;text-align:center;background:#f4f7fb;border-left:4px solid var(--navy);padding:7px 10px;margin:8px 0}
table.imgpair{border:none}table.imgpair td{border:none;width:50%;text-align:center}table.imgpair img{max-height:7.2cm}
.cap{font-size:8.5pt;color:#5b6670;margin:0;text-align:center}
.tag{font:bold 6.8pt Helvetica,Arial,sans-serif;padding:1px 5px;border-radius:3px;color:#fff;white-space:nowrap;vertical-align:1px}
.tag.v{background:#2d6a2d}.tag.m{background:#12355b}.tag.e{background:#8a6d00}.tag.i{background:#6b4a9e}.tag.u{background:#a63a1b}
.cover{page-break-after:always;padding-top:55mm}
.cover-kicker{font:600 11pt Helvetica,Arial,sans-serif;color:#5b6670;letter-spacing:.04em;text-transform:uppercase}
.cover-title{font-size:34pt;line-height:1.1;border:none;margin:10px 0;padding:0;page-break-before:avoid}
.cover-sub{font:15pt Helvetica,Arial,sans-serif;color:var(--navy);margin-bottom:22px}
.cover-meta{font-size:10pt;line-height:1.7;border-top:3px solid var(--navy);padding-top:10px;margin-bottom:18px}
.cover-box{background:var(--soft);border-left:4px solid var(--navy);padding:8px 12px;margin:10px 0;font-size:9.5pt}
.legend .tag{margin-left:6px}
.toc{page-break-after:always}.toc h1{page-break-before:avoid}
.toc table{border:none;font-size:10pt}.toc td{border:none;border-bottom:1px dotted #aab;padding:2px 4px}.toc td.n{text-align:right;width:30px}
"""


def embed(html):
    def repl(m):
        p = HERE / m.group(1)
        return 'src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"'
    return re.sub(r'src="((?:figures|guide_assets)/[^"]+)"', repl, html)


def render(pages=None):
    text = SRC.read_text(encoding="utf-8")
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
    html = re.sub(r"<h1>((?:12\.|13\.|Appendix [B-H]\.).*?)</h1>", lambda m: "<h1 class='cont'>" + m.group(1) + "</h1>", html)   # short sections flow on without a page break
    heads = re.findall(r"<h1(?: class='cont')?>(.*?)</h1>", html)
    rows = "".join(f"<tr><td>{h}</td><td class='n'>{pages[i] if pages else ''}</td></tr>" for i, h in enumerate(heads))
    toc = f"<div class='toc'><h1>Contents</h1><table>{rows}</table></div>"
    html = html.replace('<div class="toc-placeholder"></div>', toc)
    html = re.sub(r"(<img alt=\"([^\"]*)\"[^>]*>)(?!<p class)", lambda m: m.group(1) + (f"<p class='cap'>Figure: {m.group(2)}</p>" if "imgpair" not in m.group(0) else ""), html)
    html = embed(html)
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>Student Placement Probability Predictor - Complete Project Guide</title><style>{CSS}</style></head><body>{html}</body></html>"
    HTML.write_text(page, encoding="utf-8")
    return heads


def to_pdf():
    exe = next(b for b in BROWSERS if Path(b).exists())
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([exe, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--no-pdf-header-footer",
                        f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, timeout=240)


def chapter_pages(heads):
    doc = fitz.open(PDF)
    out = []
    for h in heads:
        label = re.sub(r"<[^>]+>", "", h).strip()
        hit = ""
        for i in range(2, len(doc)):                      # skip cover and contents
            if any(label == ln.strip() for ln in doc[i].get_text().splitlines()):
                hit = i + 1
                break
        out.append(hit)
    return out


def main():
    heads = render()
    to_pdf()
    pages = chapter_pages(heads)
    render(pages)
    to_pdf()
    doc = fitz.open(PDF)
    print(f"{PDF.name}: {len(doc)} pages; chapter pages: {pages}")


if __name__ == "__main__":
    main()

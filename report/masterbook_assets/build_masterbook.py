"""Builds the Learning Masterbook: parts in masterbook_assets/src/part_*.md  ->  one Markdown file  ->  HTML  ->  PDF (headless Edge/Chrome).

Presentation only. It reads no data, model or app file, and it writes only the three Masterbook files in report/.
Usage (any folder):  python report/masterbook_assets/build_masterbook.py
Needs: markdown, pymupdf, and Microsoft Edge or Google Chrome for printing. Two passes: pass 1 finds each chapter's page; pass 2 writes them into the contents.
"""
import base64
import re
import subprocess
import tempfile
from pathlib import Path

import markdown
import pymupdf

ASSETS = Path(__file__).resolve().parent
REPORT = ASSETS.parent
NAME = "Student_Placement_Predictor_Learning_Masterbook"
MD, HTML, PDF = REPORT / f"{NAME}.md", REPORT / f"{NAME}.html", REPORT / f"{NAME}.pdf"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
FLOW_ON = ("Formula Sheet", "Appendix B", "Appendix C")           # short sections that continue on the same page

CSS = """
:root{--navy:#12355b;--line:#c9d3df;--soft:#eef2f7;--orange:#D55E00}
@page{size:A4;margin:16mm 15mm 18mm 15mm;
  @bottom-left{content:"Student Placement Predictor \\00B7  Learning Masterbook";font:8pt Helvetica,Arial,sans-serif;color:#7a848d}
  @bottom-right{content:counter(page);font:9pt Helvetica,Arial,sans-serif;color:#5b6670}}
@page :first{@bottom-left{content:""}@bottom-right{content:""}}
*{box-sizing:border-box}
body{font-family:Georgia,'Times New Roman',serif;font-size:9.9pt;line-height:1.38;color:#1b1b1b;margin:0}
h1,h2,h3{font-family:Helvetica,Arial,sans-serif;color:var(--navy)}
h1{font-size:19pt;border-bottom:2px solid var(--navy);padding-bottom:4px;margin:0 0 10px;page-break-before:always}
h1.cont{page-break-before:avoid;margin-top:20px}
h2{font-size:12.5pt;margin:15px 0 5px;border-bottom:1px solid var(--line);padding-bottom:2px;page-break-after:avoid}
h3{font-size:10.5pt;margin:10px 0 3px;page-break-after:avoid}
p{margin:5px 0}
table{border-collapse:collapse;margin:8px 0;font-size:8.6pt;width:100%;page-break-inside:auto;caption-side:top}
caption{text-align:right;font:italic 7.5pt Helvetica,Arial,sans-serif;color:#7a848d;padding-bottom:1px}
tr{page-break-inside:avoid}
th,td{border:1px solid var(--line);padding:3px 6px;vertical-align:top;text-align:left}
th{background:var(--soft);font-family:Helvetica,Arial,sans-serif}
code{background:#f1f1f1;padding:0 2px;font-size:.87em;word-break:break-word}
pre{background:#f5f6f8;border-left:3px solid var(--navy);padding:6px 9px;margin:7px 0;font-size:8pt;line-height:1.3;white-space:pre-wrap;word-break:break-word;page-break-inside:avoid}
pre code{background:none;padding:0;font-size:inherit}
ul,ol{margin:4px 0 4px 18px;padding:0}li{margin:2px 0}
img{max-width:100%;display:block;margin:6px auto 0;max-height:8.6cm;width:auto}
.cap{font-size:8.3pt;color:#5b6670;margin:2px 0 8px;text-align:center;font-family:Helvetica,Arial,sans-serif}
table.imgpair{border:none}table.imgpair td{border:none;width:50%;text-align:center}table.imgpair img{max-height:6.4cm}
.eq{font-family:'Cambria Math',Cambria,Georgia,serif;font-size:11.3pt;text-align:center;background:#f4f7fb;border-left:4px solid var(--navy);padding:7px 10px;margin:8px 0;page-break-inside:avoid}
.note{background:#fff8ef;border-left:4px solid var(--orange);padding:6px 11px;margin:9px 0;font-size:9.3pt;page-break-inside:avoid}
.note p{margin:3px 0}
.wrap{background:var(--soft);border:1px solid var(--line);border-top:3px solid var(--navy);padding:6px 11px;margin:14px 0 4px;font-size:9pt;page-break-inside:avoid}
.wrap p{margin:3px 0}
.tag{font:bold 6.6pt Helvetica,Arial,sans-serif;padding:1px 5px;border-radius:3px;color:#fff;white-space:nowrap;vertical-align:1px}
.tag.v{background:#2d6a2d}.tag.m{background:#12355b}.tag.c{background:#0b7a75}.tag.s{background:#6b4a9e}.tag.e{background:#8a6d00}.tag.u{background:#a63a1b}
.cover{page-break-after:always;padding-top:42mm}
.cover-kicker{font:600 11pt Helvetica,Arial,sans-serif;color:#5b6670;letter-spacing:.04em;text-transform:uppercase}
.cover-title{font-size:34pt;line-height:1.1;border:none;margin:10px 0;padding:0;page-break-before:avoid}
.cover-sub{font:14pt Helvetica,Arial,sans-serif;color:var(--navy);margin-bottom:22px;line-height:1.3}
.cover-meta{font-size:10pt;line-height:1.7;border-top:3px solid var(--navy);padding-top:10px;margin-bottom:18px}
.cover-box{background:var(--soft);border-left:4px solid var(--navy);padding:8px 12px;margin:10px 0;font-size:9.5pt}
.legend .tag{margin-left:6px}
.toc{page-break-after:always}.toc h1{page-break-before:avoid}
.toc table{border:none;font-size:10pt}.toc td{border:none;border-bottom:1px dotted #aab;padding:2px 4px}.toc td.n{text-align:right;width:30px}
"""


def combine():
    parts = sorted((ASSETS / "src").glob("part_*.md"))
    text = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in parts) + "\n"
    # in the wrap-up boxes every line is its own paragraph
    text = re.sub(r'(<div class="wrap" markdown="1">\n)(.*?)(\n</div>)', lambda m: m.group(1) + m.group(2).replace("\n", "\n\n") + m.group(3), text, flags=re.S)
    text = re.sub(r"\n(\*\*Q\d+\.)", r"\n\n\1", text)           # one viva question per paragraph
    MD.write_text(text, encoding="utf-8")
    return text


def embed(html):
    def repl(m):
        p = REPORT / m.group(1)
        return 'src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"'
    return re.sub(r'src="(masterbook_assets/img/[^"]+)"', repl, html)


def render(text, pages=None):
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
    for key in FLOW_ON:
        html = re.sub(rf"<h1>({re.escape(key)}.*?)</h1>", lambda m: "<h1 class='cont'>" + m.group(1) + "</h1>", html)
    heads = re.findall(r"<h1(?: class='cont')?>(.*?)</h1>", html)
    rows = "".join(f"<tr><td>{h}</td><td class='n'>{pages[i] if pages else ''}</td></tr>" for i, h in enumerate(heads))
    html = html.replace('<div class="toc-placeholder"></div>', f"<div class='toc'><h1>Contents</h1><table>{rows}</table></div>")
    # figure captions: numbered, for every stand-alone image with alt text
    n = {"fig": 0}

    def cap(m):
        alt = m.group(2)
        if not alt:
            return m.group(1)
        n["fig"] += 1
        return m.group(1) + f"<p class='cap'>Figure {n['fig']}. {alt}</p>"
    html = re.sub(r"(<p><img alt=\"([^\"]*)\"[^>]*></p>)", cap, html)
    # the two confusion matrices side by side
    pair = re.search(r"(<p><img alt=\"Test confusion matrices at 0\.50 and 0\.35\"[^>]*></p>)\s*(<p><img alt=\"\"[^>]*></p>)", html)
    if pair:
        a = re.search(r"<img[^>]*>", pair.group(1)).group(0)
        b = re.search(r"<img[^>]*>", pair.group(2)).group(0)
        n["fig"] += 1
        html = html.replace(pair.group(0), f"<table class='imgpair'><tr><td>{a}<p class='cap'>threshold 0.50</p></td><td>{b}<p class='cap'>threshold 0.35</p></td></tr></table>"
                                             f"<p class='cap'>Figure {n['fig']}. Test confusion matrices (rows actual, columns predicted)</p>")
    # table numbers (not for the picture pair, contents or cover tables)
    k = {"t": 0}

    def tab(m):
        k["t"] += 1
        return f"<table><caption>Table {k['t']}</caption>"
    html = re.sub(r"<table>(?!<caption)", tab, html)
    html = html.replace("<h1>Contents</h1><table><caption>Table 1</caption>", "<h1>Contents</h1><table>")
    html = re.sub(r"(<div class='toc'><table>)<caption>Table \d+</caption>", r"", html)
    html = embed(html)
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Student Placement Predictor - Learning Masterbook</title>"
            f"<style>{CSS}</style></head><body>{html}</body></html>")
    HTML.write_text(page, encoding="utf-8")
    return heads


def to_pdf():
    exe = next(b for b in BROWSERS if Path(b).exists())
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([exe, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--no-pdf-header-footer",
                        f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, timeout=300)


def chapter_pages(heads):
    doc = pymupdf.open(PDF)
    out = []
    for h in heads:
        label = re.sub(r"<[^>]+>", "", h).strip()
        hit = ""
        for i in range(2, len(doc)):
            lines = [ln.strip() for ln in doc[i].get_text().splitlines()]
            if label in lines or any(ln and label.startswith(ln) and len(ln) > 18 for ln in lines[:6]):
                hit = i + 1
                break
        out.append(hit)
    return out


def main():
    text = combine()
    heads = render(text)
    to_pdf()
    pages = chapter_pages(heads)
    render(text, pages)
    to_pdf()
    doc = pymupdf.open(PDF)
    print(f"{PDF.name}: {len(doc)} pages, {PDF.stat().st_size/1e6:.2f} MB; HTML {HTML.stat().st_size/1e6:.2f} MB; MD {MD.stat().st_size/1e3:.0f} KB")
    print("chapter pages:", pages)


if __name__ == "__main__":
    main()

"""Render report/final_report.md to a self-contained HTML file (images embedded).

Usage (from the project root):
    python report/export_html.py

Open report/final_report.html in a browser and use Print -> Save as PDF to get a PDF.
Only presentation is done here; no analysis is run and no data files are read.
"""
import base64
import re
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SRC = HERE / "final_report.md"
OUT = HERE / "final_report.html"

CSS = """
body{font-family:Georgia,'Times New Roman',serif;line-height:1.5;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1b1b1b}
h1,h2,h3{font-family:Helvetica,Arial,sans-serif;color:#12355b}
h2{border-bottom:1px solid #c9d3df;padding-bottom:.2rem;margin-top:2.2rem}
table{border-collapse:collapse;margin:1rem 0;font-size:.88rem;width:100%}
th,td{border:1px solid #c9d3df;padding:4px 8px;vertical-align:top}
th{background:#eef2f7}
img{max-width:100%;display:block;margin:1rem auto .2rem}
blockquote{border-left:4px solid #12355b;background:#f4f7fb;margin:1rem 0;padding:.4rem 1rem}
code{background:#f1f1f1;padding:0 3px}
@page{size:A4;margin:1.5cm}
@media print{body{font-size:10pt;line-height:1.35;max-width:none;margin:0}table{font-size:8.5pt}img{max-height:7.5cm;width:auto;max-width:100%}h2{margin-top:1.2rem}img,table,blockquote{page-break-inside:avoid}p{margin:.4rem 0}}
"""


def protect_math(text):
    store = []

    def keep(m):
        store.append(m.group(0))
        return f"@@MATH{len(store) - 1}@@"

    text = re.sub(r"\$\$.+?\$\$", keep, text, flags=re.S)
    text = re.sub(r"(?<!\$)\$(?!\$)[^$\n]+?\$(?!\$)", keep, text)
    return text, store


def embed_images(html):
    def repl(m):
        path = HERE / m.group(1)
        data = base64.b64encode(path.read_bytes()).decode()
        return f'src="data:image/png;base64,{data}"'

    return re.sub(r'src="(figures/[^"]+)"', repl, html)


def main():
    text, store = protect_math(SRC.read_text(encoding="utf-8"))
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
    for i, m in enumerate(store):
        html = html.replace(f"@@MATH{i}@@", m)
    # a figure caption is the image alt text; show it under the image
    html = re.sub(r'(<img alt="([^"]*)"[^>]*>)', r'\1<p><em>\2</em></p>', html)
    html = embed_images(html)
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Predicting Student Placement Probability - Final Report</title>"
        f"<style>{CSS}</style>"
        "<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]}};</script>"
        "<script async src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script>"
        f"</head><body>{html}</body></html>"
    )
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

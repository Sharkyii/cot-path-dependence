import markdown, subprocess, os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = f"{REPO_ROOT}/paper_build"

with open(f"{REPO_ROOT}/PAPER_DRAFT.md") as f:
    md = f.read()

# Drop the working-draft preamble (source-pointer note) -- not part of the paper itself
md = md.split("---", 1)[1]

fig2 = """<div class="figure">
<img src="figs/fig2_stratum_pvalues.png" alt="Fig 2">
<p class="caption"><strong>Figure 1.</strong> Fisher's combined p-value by difficulty stratum and overall. No stratum, nor the overall combined test, reaches the α = 0.05 threshold (dashed line).</p>
</div>"""

fig1 = """<div class="figure">
<img src="figs/fig1_pdi_distribution.png" alt="Fig 1">
<p class="caption"><strong>Figure 2.</strong> PDI across all 21 matched pairs, sorted. 16/21 pairs show exactly PDI = 0. Of the 5 nonzero pairs, only one (the tallest bar) survives Bonferroni correction across the 21 tests performed.</p>
</div>"""

fig3 = """<div class="figure">
<img src="figs/fig3_decisiveness_followup.png" alt="Fig 3">
<p class="caption"><strong>Figure 3.</strong> The two flagged anomalies' decisiveness gap (% of branches ending in a stated, non-empty answer) at L1, and the exploratory L2/L3 follow-up (n=1 pair per level per problem) where the gap closes.</p>
</div>"""

def insert_before(md, anchor, figure_md):
    idx = md.index(anchor)
    return md[:idx] + figure_md + "\n\n" + md[idx:]

md = insert_before(md, "### 4.2 Distribution of individual-pair results", fig2)
md = insert_before(md, "### 4.3 Characterizing the two real anomalies", fig1)
md = insert_before(md, "## 5. Limitations and Future Work", fig3)

body = markdown.markdown(md, extensions=["tables", "fenced_code"])

# Split off Abstract (+ the H1/H2 that introduce it) to render full-width above
# the two-column body, matching the NeurIPS/ACL/ICML convention.
marker = '<h2>Abstract</h2>'
pre, _, rest = body.partition(marker)
abstract_p, _, columned = rest.partition('</p>')
abstract_html = f'<div class="abstract"><strong>Abstract</strong><p>{abstract_p.split("<p>",1)[1]}</p></div>'

TITLE = "On the Sufficiency of Observable State for Chain-of-Thought Early Stopping"

html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{TITLE}</title>
<style>
  @page {{ size: A4; margin: 20mm 15mm 22mm 15mm; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 9.8pt;
    line-height: 1.42;
    color: #111;
    margin: 0;
  }}
  .titleblock {{ text-align: center; margin-bottom: 14px; }}
  .titleblock h1 {{
    font-size: 17pt; font-weight: bold; margin: 0 0 8px 0; line-height: 1.25;
  }}
  .authors {{ font-size: 10.5pt; margin-bottom: 2px; }}
  .affil {{ font-size: 9pt; color: #444; font-style: italic; }}
  .abstract {{
    max-width: 92%; margin: 14px auto 20px auto; font-size: 9.5pt;
    line-height: 1.4;
  }}
  .abstract strong {{
    display: block; text-align: center; font-size: 10.5pt; margin-bottom: 4px;
  }}
  .columns {{
    column-count: 2; column-gap: 8mm; column-rule: 0.5px solid #ddd;
    text-align: justify;
  }}
  h2 {{
    font-size: 11pt; margin: 14px 0 6px 0; break-after: avoid-column;
    border-bottom: 0.75px solid #333; padding-bottom: 2px;
  }}
  h3 {{ font-size: 10pt; margin: 10px 0 4px 0; break-after: avoid-column; }}
  p {{ margin: 0 0 7px 0; text-indent: 0; }}
  code {{ background: #f2f2f2; padding: 0 3px; border-radius: 2px; font-size: 8.6pt; }}
  table {{
    border-collapse: collapse; margin: 8px auto; font-size: 8.3pt; width: 100%;
    break-inside: avoid;
  }}
  th, td {{ border: 0.75px solid #999; padding: 3px 5px; text-align: center; }}
  th {{ background: #eee; }}
  .figure {{
    column-span: all; text-align: center; margin: 12px 0; break-inside: avoid;
  }}
  .figure img {{ max-width: 78%; }}
  .caption {{
    font-size: 8.3pt; text-align: left; color: #222; margin: 4px auto 0 auto;
    max-width: 78%; line-height: 1.35;
  }}
  hr {{ display: none; }}
  ul, ol {{ margin: 4px 0 8px 18px; padding: 0; }}
  li {{ margin: 2px 0; }}
  strong {{ font-weight: bold; }}
  em {{ font-style: italic; }}
</style>
</head>
<body>
<div class="titleblock">
  <h1>{TITLE}</h1>
  <div class="authors">Sneh Kansagara</div>
  <div class="affil">ABV-Indian Institute of Information Technology and Management, Gwalior &middot; snehkansagara@gmail.com</div>
</div>
{abstract_html}
<div class="columns">
{columned}
</div>
</body>
</html>
"""

with open(f"{BUILD}/paper.html", "w") as f:
    f.write(html)

print("HTML written:", os.path.getsize(f"{BUILD}/paper.html"), "bytes")

pdf_path = f"{REPO_ROOT}/PAPER_DRAFT.pdf"
result = subprocess.run(
    ["google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
     f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer",
     f"file://{BUILD}/paper.html"],
    capture_output=True, text=True, timeout=60,
)
print("chrome stdout:", result.stdout[-500:])
print("chrome stderr:", result.stderr[-1500:])
print("PDF exists:", os.path.exists(pdf_path), os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0)

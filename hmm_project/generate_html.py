"""
Convert report.md to a standalone report.html with MathJax for equation rendering.
Open the HTML in Chrome and Ctrl+P -> Save as PDF for a perfect result.
"""

import os
import re
import markdown

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def protect_math(md_text):
    """
    Temporarily replace $...$ and $$...$$ blocks with placeholders
    so the Markdown parser doesn't mangle underscores inside math.
    """
    placeholders = []

    # Display math first ($$...$$), including multi-line
    def replace_display(m):
        placeholders.append(m.group(0))
        return f"\x00MATH{len(placeholders)-1}\x00"
    md_text = re.sub(r'\$\$.+?\$\$', replace_display, md_text, flags=re.DOTALL)

    # Inline math ($...$)
    def replace_inline(m):
        placeholders.append(m.group(0))
        return f"\x00MATH{len(placeholders)-1}\x00"
    md_text = re.sub(r'\$(?!\$)(.+?)\$', replace_inline, md_text)

    return md_text, placeholders

def restore_math(html_text, placeholders):
    """Put the math expressions back after Markdown conversion."""
    for i, original in enumerate(placeholders):
        html_text = html_text.replace(f"\x00MATH{i}\x00", original)
    return html_text

def convert():
    md_path = os.path.join(SCRIPT_DIR, "report.md")
    html_path = os.path.join(SCRIPT_DIR, "report.html")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Protect math from Markdown parser
    md_text, placeholders = protect_math(md_text)

    # Convert Markdown to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br"],
    )

    # Restore math expressions
    html_body = restore_math(html_body, placeholders)

    # Convert relative image paths to absolute file:// URIs so they load in the browser
    def abs_img(m):
        src = m.group(1)
        if not src.startswith("http") and not src.startswith("file"):
            abs_path = os.path.abspath(os.path.join(SCRIPT_DIR, src)).replace("\\", "/")
            src = f"file:///{abs_path}"
        return f'src="{src}"'
    html_body = re.sub(r'src="([^"]+)"', abs_img, html_body)

    # Wrap in full HTML with MathJax and print-friendly CSS
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HMM Supply Chain Disruption Detection - Project Report</title>

<!-- MathJax for LaTeX equation rendering -->
<script>
MathJax = {{
  tex: {{
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']],
  }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>

<style>
  body {{
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 12pt;
    line-height: 1.7;
    max-width: 8in;
    margin: 0 auto;
    padding: 0.75in 1in;
    color: #222;
  }}
  h1 {{
    font-size: 20pt;
    text-align: center;
    border-bottom: 2px solid #333;
    padding-bottom: 12px;
    margin-top: 0;
  }}
  h2 {{
    font-size: 15pt;
    border-bottom: 1px solid #aaa;
    padding-bottom: 6px;
    margin-top: 35px;
  }}
  h3 {{
    font-size: 12.5pt;
    margin-top: 22px;
  }}
  table {{
    border-collapse: collapse;
    margin: 18px auto;
    font-size: 10.5pt;
  }}
  th, td {{
    border: 1px solid #aaa;
    padding: 6px 14px;
    text-align: left;
  }}
  th {{
    background-color: #f0f0f0;
    font-weight: bold;
  }}
  img {{
    max-width: 100%;
    display: block;
    margin: 18px auto;
  }}
  code {{
    background-color: #f5f5f5;
    padding: 1px 5px;
    font-size: 10pt;
    font-family: Consolas, 'Courier New', monospace;
    border-radius: 3px;
  }}
  pre {{
    background-color: #f5f5f5;
    padding: 12px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 10pt;
  }}
  hr {{
    border: none;
    border-top: 1px solid #ccc;
    margin: 25px 0;
  }}
  blockquote {{
    border-left: 3px solid #ccc;
    margin-left: 0;
    padding-left: 15px;
    color: #555;
  }}

  /* Print-friendly styles for Ctrl+P -> Save as PDF */
  @media print {{
    body {{
      margin: 0;
      padding: 0.5in;
      font-size: 11pt;
    }}
    h1 {{ font-size: 18pt; }}
    h2 {{ font-size: 14pt; page-break-after: avoid; }}
    h3 {{ page-break-after: avoid; }}
    img {{ page-break-inside: avoid; max-width: 100%; }}
    table {{ page-break-inside: avoid; }}
    pre {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML saved to: {html_path}")
    print()
    print("To create the PDF:")
    print("  1. Double-click report.html to open in Chrome")
    print("  2. Wait a moment for MathJax to render the equations")
    print("  3. Ctrl+P -> Destination: 'Save as PDF' -> Save")

if __name__ == "__main__":
    convert()

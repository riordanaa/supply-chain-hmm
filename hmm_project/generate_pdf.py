"""
Generate a PDF report from report.md with embedded images.
Uses fpdf2 which has no system dependencies.
"""

import os
import re
from fpdf import FPDF

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Unicode -> ASCII replacements for Helvetica font
UNICODE_MAP = {
    '\u2014': '--',    # em dash
    '\u2013': '-',     # en dash
    '\u2018': "'",     # left single quote
    '\u2019': "'",     # right single quote
    '\u201c': '"',     # left double quote
    '\u201d': '"',     # right double quote
    '\u2026': '...',   # ellipsis
    '\u2192': '->',    # right arrow
    '\u2264': '<=',    # less than or equal
    '\u2265': '>=',    # greater than or equal
    '\u03b1': 'alpha', # alpha
    '\u03b2': 'beta',  # beta
    '\u03c0': 'pi',    # pi
    '\u03bb': 'lambda',# lambda
    '\u03c3': 'sigma', # sigma
    '\u03c7': 'chi',   # chi
    '\u2248': '~=',    # approximately
    '\u00d7': 'x',     # multiplication sign
    '\u2212': '-',     # minus sign
    '\u2260': '!=',    # not equal
    '\u221e': 'inf',   # infinity
    '\u2211': 'Sum',   # summation
    '\u220f': 'Prod',  # product
    '\u2208': 'in',    # element of
    '\u2032': "'",     # prime
    '\u00b2': '2',     # superscript 2
    '\u2113': 'l',     # script l
    '\u0302': '',      # combining circumflex
    '\u0304': '',      # combining macron
    '\u0308': '',      # combining dieresis
    '\u2103': 'C',     # degree celsius
    '\xb1': '+/-',     # plus minus
    '\xb7': '.',       # middle dot
    chr(8226): '-',    # bullet
}


def sanitize(text):
    """Replace unicode characters with ASCII equivalents."""
    for uc, asc in UNICODE_MAP.items():
        text = text.replace(uc, asc)
    # Catch any remaining non-latin1 characters
    result = []
    for ch in text:
        try:
            ch.encode('latin-1')
            result.append(ch)
        except UnicodeEncodeError:
            result.append('?')
    return ''.join(result)


def strip_md(text):
    """Remove markdown formatting markers."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\$\$(.+?)\$\$', r'\1', text)
    text = re.sub(r'\$(.+?)\$', r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # links
    return sanitize(text)


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def build_pdf(md_path, output_path):
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pdf = ReportPDF("P", "mm", "Letter")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin
    LH = 5.5

    in_table = False
    table_rows = []
    in_code = False
    code_lines = []

    def flush_table():
        nonlocal table_rows, in_table
        data = []
        for row_str in table_rows:
            cells = [sanitize(c.strip()) for c in row_str.strip("|").split("|")]
            if all(re.match(r'^[-:]+$', c) for c in cells):
                continue
            data.append(cells)
        if data:
            nc = max(len(r) for r in data)
            cw = W / nc
            for i, row in enumerate(data):
                pdf.set_font("Helvetica", "B" if i == 0 else "", 9)
                for j in range(nc):
                    cell_text = row[j] if j < len(row) else ""
                    pdf.cell(cw, 6, cell_text[:60], border=1, align="C" if i == 0 else "L")
                pdf.ln()
            pdf.ln(3)
        table_rows = []
        in_table = False

    def flush_code():
        nonlocal code_lines, in_code
        pdf.set_font("Courier", "", 8)
        pdf.set_fill_color(245, 245, 245)
        for cl in code_lines:
            pdf.cell(W, 4, sanitize(cl[:130]), new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11)
        code_lines = []
        in_code = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")

        # Code fences
        if line.startswith("```"):
            if in_code:
                flush_code()
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # Table
        if "|" in line and line.strip().startswith("|"):
            in_table = True
            table_rows.append(line)
            i += 1
            continue
        elif in_table:
            flush_table()

        # Blank
        if not line.strip():
            pdf.ln(3)
            i += 1
            continue

        # HR
        if line.strip() == "---":
            pdf.ln(2)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)
            i += 1
            continue

        # H1
        if line.startswith("# ") and not line.startswith("##"):
            pdf.set_font("Helvetica", "B", 18)
            pdf.ln(5)
            pdf.multi_cell(W, 9, strip_md(line[2:]), align="C")
            pdf.ln(3)
            pdf.set_font("Helvetica", "", 11)
            i += 1
            continue

        # H2
        if line.startswith("## "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.ln(5)
            pdf.cell(W, 8, strip_md(line[3:]), new_x="LMARGIN", new_y="NEXT")
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
            pdf.set_font("Helvetica", "", 11)
            i += 1
            continue

        # H3
        if line.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.ln(3)
            pdf.cell(W, 7, strip_md(line[4:]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            pdf.set_font("Helvetica", "", 11)
            i += 1
            continue

        # Image
        img_match = re.match(r'!\[(.+?)\]\((.+?)\)', line.strip())
        if img_match:
            caption = img_match.group(1)
            img_path = os.path.join(os.path.dirname(md_path), img_match.group(2))
            if os.path.exists(img_path):
                if pdf.get_y() > 170:
                    pdf.add_page()
                pdf.image(img_path, x=pdf.l_margin, w=W)
                pdf.ln(2)
                pdf.set_font("Helvetica", "I", 9)
                pdf.multi_cell(W, 4.5, sanitize(caption), align="C")
                pdf.ln(3)
                pdf.set_font("Helvetica", "", 11)
            i += 1
            continue

        # Italic caption
        if line.strip().startswith("*") and line.strip().endswith("*") and len(line.strip()) > 2:
            text = strip_md(line.strip().strip("*"))
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(W, 4.5, text, align="C")
            pdf.ln(2)
            pdf.set_font("Helvetica", "", 11)
            i += 1
            continue

        # Display math
        if line.strip().startswith("$$"):
            math = line.strip().strip("$").strip()
            if not line.strip().endswith("$$") or line.strip() == "$$":
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    math += " " + lines[i].strip()
                    i += 1
                if i < len(lines):
                    math += " " + lines[i].strip().rstrip("$").strip()
            pdf.set_font("Courier", "", 10)
            pdf.ln(2)
            pdf.multi_cell(W, 5, "  " + sanitize(math), align="C")
            pdf.ln(2)
            pdf.set_font("Helvetica", "", 11)
            i += 1
            continue

        # Bullet
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            text = strip_md(line.strip()[2:])
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(5, LH, "-")
            pdf.multi_cell(W - 5, LH, text)
            i += 1
            continue

        # Numbered list
        num_match = re.match(r'^(\d+)\.\s+(.+)', line.strip())
        if num_match:
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(8, LH, f"{num_match.group(1)}.")
            pdf.multi_cell(W - 8, LH, strip_md(num_match.group(2)))
            i += 1
            continue

        # Regular paragraph
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(W, LH, strip_md(line.strip()))
        i += 1

    if in_table:
        flush_table()

    pdf.output(output_path)
    print(f"PDF saved to: {output_path}")


if __name__ == "__main__":
    md_path = os.path.join(os.path.dirname(__file__), "report.md")
    out_path = os.path.join(os.path.dirname(__file__), "report.pdf")
    build_pdf(md_path, out_path)

"""Builds the master's thesis .docx from thesis/text/*.txt (formatting per LNU guidelines, 2023).

    python thesis/build_docx.py [--pages N]

Markup (one paragraph per line):
    @title, @task, @toc, @refs, @pagebreak      special blocks
    @struct НАЗВА                                unnumbered structural heading (new page)
    @chapter N | НАЗВА                           chapter heading "РОЗДІЛ N НАЗВА" (new page)
    ## 1.1 Назва / ### 1.1.1 Назва              subsection / point
    @fig path | width_cm | Рисунок N – caption
    @table Таблиця N – caption | w1,w2,...      followed by "| a | b |" rows (first = header)
    @eq formula | (N)
    @abbrev ... @endabbrev                       "term | explanation" rows
    @appendix А | Назва
    @sourcecode path | N | caption | start | end  code listing from the repository
    - item                                       list item
    **text**                                     bold run; [@key; @key2] citations
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from references import REFERENCES

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "thesis"
OUT = THESIS / "Магістерська_робота.docx"
FONT = "Times New Roman"
TEXT_WIDTH_CM = 17.5
NO_TOC = {"АНОТАЦІЯ", "ABSTRACT"}

TOPIC = "Мультимодальна система класифікації об'єктів дорожньої сцени на основі Transformer-злиття даних RGB-камери та LiDAR"


# ----------------------------------------------------------------------------- helpers


def set_cell_margins(cell, top=40, bottom=40, left=80, right=80):
    tc_pr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for side, value in (("top", top), ("bottom", bottom), ("start", left), ("end", right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        mar.append(node)
    tc_pr.append(mar)


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:tblHeader")
    node.set(qn("w:val"), "true")
    tr_pr.append(node)


def cant_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))


def add_field(paragraph, instr: str, placeholder: str = ""):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    run._r.append(begin)
    run = paragraph.add_run()
    text = OxmlElement("w:instrText")
    text.set(qn("xml:space"), "preserve")
    text.text = f" {instr} "
    run._r.append(text)
    run = paragraph.add_run()
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    run._r.append(sep)
    paragraph.add_run(placeholder)
    run = paragraph.add_run()
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(end)


def no_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:val"), "nil")
        borders.append(node)
    tbl_pr.append(borders)


# ----------------------------------------------------------------------------- builder


class ThesisBuilder:
    def __init__(self, pages: str = "{PAGES}") -> None:
        self.doc = Document()
        self.pages = pages
        self.cite_order: list[str] = []
        self.fig_count = 0
        self.tab_count = 0
        self.app_count = 0
        self._setup_styles()

    # --- styles and page

    def _setup_styles(self) -> None:
        section = self.doc.sections[0]
        section.page_height, section.page_width = Cm(29.7), Cm(21.0)
        section.top_margin = section.bottom_margin = Cm(2.0)
        section.left_margin, section.right_margin = Cm(2.5), Cm(1.0)
        section.header_distance = Cm(1.0)

        normal = self.doc.styles["Normal"]
        normal.font.name = FONT
        normal.font.size = Pt(14)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        pf = normal.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.space_before = pf.space_after = Pt(0)
        pf.first_line_indent = Cm(1.25)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.widow_control = True

        for level, (size, center, caps) in {
            1: (14, True, True),
            2: (14, False, False),
            3: (14, False, False),
        }.items():
            style = self.doc.styles[f"Heading {level}"]
            style.font.name = FONT
            style.font.size = Pt(size)
            style.font.bold = True
            style.font.italic = False
            style.font.all_caps = caps
            style.font.color.rgb = None
            rpr = style.element.get_or_add_rPr()
            rpr.get_or_add_rFonts().set(qn("w:ascii"), FONT)
            rpr.get_or_add_rFonts().set(qn("w:hAnsi"), FONT)
            rpr.get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
            rpr.get_or_add_rFonts().set(qn("w:cs"), FONT)
            for theme_attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                rpr.get_or_add_rFonts().attrib.pop(qn(theme_attr), None)
            hp = style.paragraph_format
            hp.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
            hp.first_line_indent = Cm(0) if center else Cm(1.25)
            hp.space_before = Pt(0 if level == 1 else 21)
            hp.space_after = Pt(21)
            hp.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            hp.keep_with_next = True

        # TOC entries: same font, dot leaders by default in Word.
        for level in (1, 2, 3):
            name = f"toc {level}"
            try:
                style = self.doc.styles[name]
            except KeyError:
                from docx.enum.style import WD_STYLE_TYPE

                style = self.doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            style.base_style = self.doc.styles["Normal"]
            style.font.name = FONT
            style.font.size = Pt(14)
            sp = style.paragraph_format
            sp.first_line_indent = Cm(0)
            sp.left_indent = Cm(0.6 * (level - 1))
            sp.alignment = WD_ALIGN_PARAGRAPH.LEFT
            sp.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    # --- inline text

    def _cite(self, match: re.Match) -> str:
        keys = [k.strip().lstrip("@") for k in match.group(1).split(";")]
        nums = []
        for key in keys:
            if key not in REFERENCES:
                raise KeyError(f"unknown reference {key}")
            if key not in self.cite_order:
                self.cite_order.append(key)
            nums.append(self.cite_order.index(key) + 1)
        nums = sorted(set(nums))
        parts, start = [], nums[0]
        for prev, cur in zip(nums, nums[1:] + [None]):
            if cur != prev + 1:
                parts.append(
                    str(start)
                    if start == prev
                    else (f"{start}, {prev}" if prev == start + 1 else f"{start}–{prev}")
                )
                start = cur
        return "[" + ", ".join(parts) + "]"

    def _inline(self, paragraph, text: str, size: int | None = None) -> None:
        text = re.sub(r"\[(@[^\]]+)\]", self._cite, text)
        for i, chunk in enumerate(re.split(r"\*\*", text)):
            if not chunk:
                continue
            for piece in re.split(r"(_\{[^}]*\}|\^\{[^}]*\})", chunk):
                if not piece:
                    continue
                if piece.startswith("_{"):
                    run = paragraph.add_run(piece[2:-1])
                    run.font.subscript = True
                elif piece.startswith("^{"):
                    run = paragraph.add_run(piece[2:-1])
                    run.font.superscript = True
                else:
                    run = paragraph.add_run(piece)
                run.bold = i % 2 == 1
                if size:
                    run.font.size = Pt(size)

    def para(
        self,
        text: str = "",
        indent: bool = True,
        align=None,
        bold=False,
        size=None,
        keep=False,
        space_after=0,
    ):
        p = self.doc.add_paragraph()
        if not indent:
            p.paragraph_format.first_line_indent = Cm(0)
        if align is not None:
            p.paragraph_format.alignment = align
        if keep:
            p.paragraph_format.keep_with_next = True
        if space_after:
            p.paragraph_format.space_after = Pt(space_after)
        if bold:
            run = p.add_run(text)
            run.bold = True
            if size:
                run.font.size = Pt(size)
        else:
            self._inline(p, text, size)
        return p

    # --- blocks

    def page_break(self) -> None:
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def heading(self, text: str, level: int, new_page: bool = False, toc: bool = True):
        if toc:
            p = self.doc.add_heading(level=level)
        else:
            p = self.doc.add_paragraph(style="Heading 1")
            p.paragraph_format.outline_level = None
            # Not in the TOC: plain paragraph formatted like a heading.
            p.style = self.doc.styles["Normal"]
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_after = Pt(21)
            p.paragraph_format.keep_with_next = True
        if new_page:
            p.paragraph_format.page_break_before = True
        run = p.add_run(text)
        if not toc:
            run.bold = True
        return p

    def title_page(self) -> None:
        c = WD_ALIGN_PARAGRAPH.CENTER
        for line in ("МІНІСТЕРСТВО ОСВІТИ І НАУКИ УКРАЇНИ", "Львівський національний університет імені Івана Франка",
                     "Факультет електроніки та комп'ютерних технологій", "Кафедра [назва кафедри]"):  # fmt: skip
            p = self.para(line, indent=False, align=c, bold=True)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        self.para()
        right = Cm(8.5)
        for text, bold in (("Допустити до захисту", True), ("Зав. кафедри", False),
                           ("_________ [звання, Ім'я ПРІЗВИЩЕ]", True), ("«___» ____________ 2026 р.", False)):  # fmt: skip
            p = self.para(text, indent=False, bold=bold)
            p.paragraph_format.left_indent = right
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.paragraph_format.space_after = Pt(4)
        for _ in range(2):
            self.para()
        self.para("Кваліфікаційна робота", indent=False, align=c, bold=True)
        p = self.para("Магістр", indent=False, align=c, bold=True)
        p.runs[0].underline = True
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        p = self.para("(освітній рівень)", indent=False, align=c, size=10)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        self.para()
        self.para(TOPIC, indent=False, align=c, bold=True)
        self.para()
        for text, bold in (("Виконав:", False), ("студент групи ФеІм-[__]", False), ("спеціальності 122 – Комп'ютерні науки", False),
                           ("", False), ("___________ [Ім'я ПРІЗВИЩЕ]", True), ("", False), ("Науковий керівник:", False),
                           ("___________ [звання, Ім'я ПРІЗВИЩЕ]", True), ("«___» ____________ 2026 р.", False), ("", False),
                           ("Рецензент:", False), ("___________ [звання, Ім'я ПРІЗВИЩЕ]", True)):  # fmt: skip
            p = self.para(text, indent=False, bold=bold)
            p.paragraph_format.left_indent = right
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        self.para()
        self.para("Львів 2026", indent=False, align=c)
        self.page_break()

    def task_page(self) -> None:
        self.heading("ЗАВДАННЯ НА МАГІСТЕРСЬКУ РОБОТУ", 1, toc=False)
        self.para("[Бланк завдання (Додаток Б методичних рекомендацій) заповнює науковий керівник з обох боків аркуша. "
                  "Його вкладають у пояснювальну записку другим аркушем замість цієї сторінки.]",
                  indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)  # fmt: skip
        self.page_break()

    def toc(self) -> None:
        self.heading("ЗМІСТ", 1, new_page=True, toc=False)
        p = self.doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        add_field(p, 'TOC \\o "1-3" \\h \\z \\u', "Оновіть зміст (F9)")
        # Numbered pages start after the table of contents.
        new = self.doc.add_section(WD_SECTION.NEW_PAGE)
        new.header.is_linked_to_previous = False
        hp = new.header.paragraphs[0]
        hp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hp.paragraph_format.first_line_indent = Cm(0)
        add_field(hp, "PAGE", "1")
        self._first_in_section = True

    def figure(self, path: str, width: float, caption: str) -> None:
        self.fig_count += 1
        p = self.doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        p.add_run().add_picture(str(THESIS / path), width=Cm(min(width, TEXT_WIDTH_CM - 0.5)))
        cap = self.para(caption, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)
        cap.paragraph_format.space_before = Pt(6)

    def table(
        self, caption: str, widths: list[float], rows: list[list[str]], font_size: int = 12
    ) -> None:
        self.tab_count += 1
        cap = self.para(caption, indent=False, align=WD_ALIGN_PARAGRAPH.LEFT, keep=True)
        cap.paragraph_format.space_before = Pt(6)
        ncols = len(rows[0])
        if len(widths) != ncols:
            widths = [TEXT_WIDTH_CM / ncols] * ncols
        table = self.doc.add_table(rows=0, cols=ncols)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        for r_i, cells in enumerate(rows):
            row = table.add_row()
            cant_split(row)
            if r_i == 0:
                repeat_header(row)
            for c_i, text in enumerate(cells):
                cell = row.cells[c_i]
                cell.width = Cm(widths[c_i])
                set_cell_margins(cell)
                p = cell.paragraphs[0]
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                numeric = bool(re.fullmatch(r"[\d\s,.–%−+·⁻⁰¹²³⁴⁵⁶⁷⁸⁹≥ ]+|–", text.strip()))
                p.paragraph_format.alignment = (
                    WD_ALIGN_PARAGRAPH.CENTER if r_i == 0 or numeric else WD_ALIGN_PARAGRAPH.LEFT
                )
                if r_i < len(rows) - 1:
                    p.paragraph_format.keep_with_next = True
                run = p.add_run(re.sub(r"\[(@[^\]]+)\]", self._cite, text.strip()))
                run.font.size = Pt(font_size)
                run.bold = r_i == 0
        for c_i, w in enumerate(widths):
            for cell in table.columns[c_i].cells:
                cell.width = Cm(w)
        self.para(space_after=0).paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

    def equation(self, formula: str, number: str) -> None:
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.first_line_indent = Cm(0)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.space_before = Pt(10)
        pf.space_after = Pt(10)
        pf.keep_with_next = True
        pf.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM / 2), WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT)
        # Linear-format (UnicodeMath) text; word_finalize.ps1 converts it into a Word equation.
        p.add_run("\t")
        p.add_run(formula)
        p.add_run("\t" + number)

    def list_item(self, text: str) -> None:
        p = self.para("– " + text)
        p.paragraph_format.first_line_indent = Cm(1.25)

    def abbreviations(self, rows: list[list[str]]) -> None:
        table = self.doc.add_table(rows=0, cols=2)
        no_borders(table)
        table.autofit = False
        for term, expl in sorted(rows, key=lambda r: r[0].lower()):
            row = table.add_row()
            for i, (text, w) in enumerate(((term, 4.0), (f"– {expl}", 13.0))):
                cell = row.cells[i]
                cell.width = Cm(w)
                p = cell.paragraphs[0]
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.add_run(text)

    def references(self) -> None:
        for i, key in enumerate(self.cite_order, start=1):
            p = self.para(f"{i}. {REFERENCES[key]}")
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        unused = sorted(set(REFERENCES) - set(self.cite_order))
        if unused:
            print("WARNING: references never cited:", unused)

    def appendix(self, letter: str, title: str) -> None:
        self.app_count += 1
        p = self.doc.add_heading(level=1)
        p.paragraph_format.page_break_before = True
        run = p.add_run(f"Додаток {letter}")
        run.font.all_caps = False
        p.add_run().add_break()
        run = p.add_run(title)
        run.font.all_caps = False

    def source_code(self, path: str, number: str, caption: str, start: str, end: str) -> None:
        lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
        if start:
            i0 = next(i for i, line in enumerate(lines) if line.startswith(start))
        else:
            i0 = 0
        i1 = len(lines)
        if end:
            i1 = next(
                i for i, line in enumerate(lines[i0 + 1 :], start=i0 + 1) if line.startswith(end)
            )
        code = lines[i0:i1]
        while code and not code[-1].strip():
            code.pop()
        cap = self.para(f"Лістинг {number} – {caption}", indent=False, keep=True)
        cap.paragraph_format.space_before = Pt(12)
        for line in code:
            p = self.doc.add_paragraph()
            pf = p.paragraph_format
            pf.first_line_indent = Cm(0)
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            run = p.add_run(line.replace("\t", "    ") or " ")
            run.font.name = "Courier New"
            run._r.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), "Courier New")
            run.font.size = Pt(9)

    # --- driver

    def build(self, files: list[Path]) -> None:
        lines: list[str] = []
        for f in files:
            lines += f.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            i += 1
            if not line:
                continue
            if line == "@title":
                self.title_page()
            elif line == "@task":
                self.task_page()
            elif line == "@toc":
                self.toc()
            elif line == "@pagebreak":
                self.page_break()
            elif line == "@refs":
                self.references()
            elif line.startswith("@struct "):
                name = line[len("@struct ") :].strip()
                self.heading(name, 1, new_page=name not in ("АНОТАЦІЯ",), toc=name not in NO_TOC)
            elif line.startswith("@chapter "):
                num, title = [s.strip() for s in line[len("@chapter ") :].split("|")]
                self.heading(f"РОЗДІЛ {num} {title}", 1, new_page=True)
            elif line.startswith("### "):
                self.heading(line[4:], 3)
            elif line.startswith("## "):
                self.heading(line[3:], 2)
            elif line.startswith("@fig "):
                path, width, caption = [s.strip() for s in line[5:].split("|")]
                self.figure(path, float(width), caption)
            elif line.startswith("@table "):
                spec = [s.strip() for s in line[7:].split("|")]
                widths = [float(w) for w in spec[1].split(",")] if len(spec) > 1 and spec[1] else []
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append([c for c in lines[i].strip().strip("|").split("|")])
                    i += 1
                self.table(spec[0], widths, rows)
            elif line.startswith("@eq "):
                formula, number = line[4:].rsplit("|", 1)
                self.equation(formula.strip(), number.strip())
            elif line == "@abbrev":
                rows = []
                while lines[i].strip() != "@endabbrev":
                    rows.append([s.strip() for s in lines[i].split("|", 1)])
                    i += 1
                i += 1
                self.abbreviations(rows)
            elif line.startswith("@appendix "):
                letter, title = [s.strip() for s in line[len("@appendix ") :].split("|")]
                self.appendix(letter, title)
            elif line.startswith("@sourcecode "):
                path, number, caption, start, end = [
                    s.strip() for s in line[len("@sourcecode ") :].split("|")
                ]
                self.source_code(path, number, caption, start, end)
            elif line.startswith("- "):
                self.list_item(line[2:])
            else:
                self.para(line)

    def fill_counts(self) -> None:
        values = {"{REFS}": str(len(self.cite_order)), "{APPS}": str(self.app_count), "{FIGS}": str(self.fig_count),
                  "{TABS}": str(self.tab_count), "{PAGES}": self.pages}  # fmt: skip
        for p in self.doc.paragraphs:
            for run in p.runs:
                for k, v in values.items():
                    if k in run.text:
                        run.text = run.text.replace(k, v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", default="{PAGES}")
    args = parser.parse_args()
    builder = ThesisBuilder(args.pages)
    builder.build(sorted((THESIS / "text").glob("*.txt")))
    builder.fill_counts()
    builder.doc.save(OUT)
    print(
        f"{OUT} | figures {builder.fig_count}, tables {builder.tab_count}, references {len(builder.cite_order)}"
    )


if __name__ == "__main__":
    main()

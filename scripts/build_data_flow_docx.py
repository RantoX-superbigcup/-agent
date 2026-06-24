"""Build docs/data_flow.docx from docs/data_flow.md without third-party packages."""

from __future__ import annotations

from pathlib import Path
import html
import re
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "docs" / "data_flow.md"
TARGET = PROJECT_ROOT / "docs" / "data_flow.docx"

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def main() -> int:
    markdown = SOURCE.read_text(encoding="utf-8")
    document_xml = build_document_xml(markdown)
    with zipfile.ZipFile(TARGET, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", content_types_xml())
        package.writestr("_rels/.rels", root_rels_xml())
        package.writestr("word/_rels/document.xml.rels", document_rels_xml())
        package.writestr("word/document.xml", document_xml)
        package.writestr("word/styles.xml", styles_xml())
        package.writestr("word/settings.xml", settings_xml())
    print(TARGET)
    return 0


def build_document_xml(markdown: str) -> str:
    body_parts: list[str] = []
    lines = markdown.splitlines()
    paragraph_buffer: list[str] = []
    code_buffer: list[str] = []
    in_code = False
    index = 0

    def flush_paragraph() -> None:
        if paragraph_buffer:
            body_parts.append(paragraph(" ".join(paragraph_buffer).strip()))
            paragraph_buffer.clear()

    def flush_code() -> None:
        if code_buffer:
            body_parts.append(code_block("\n".join(code_buffer)))
            code_buffer.clear()

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            index += 1
            continue

        if in_code:
            code_buffer.append(line)
            index += 1
            continue

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            flush_paragraph()
            level = min(len(heading_match.group(1)), 3)
            body_parts.append(paragraph(clean_inline(heading_match.group(2)), style=f"Heading{level}"))
            index += 1
            continue

        if is_table_start(lines, index):
            flush_paragraph()
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            body_parts.append(table(table_lines))
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            body_parts.append(paragraph(clean_inline(stripped[2:]), style="ListParagraph"))
            index += 1
            continue

        paragraph_buffer.append(clean_inline(stripped))
        index += 1

    flush_paragraph()
    flush_code()

    body_parts.append(
        '<w:sectPr>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="720" w:footer="720" w:gutter="0"/>'
        '</w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{NS_W}"><w:body>{"".join(body_parts)}</w:body></w:document>'
    )


def clean_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "")
    return text


def is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    current = lines[index].strip()
    next_line = lines[index + 1].strip()
    return current.startswith("|") and next_line.startswith("|") and "---" in next_line


def paragraph(text: str, style: str = "Normal") -> str:
    safe_text = html.escape(text)
    return (
        "<w:p>"
        f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
        "<w:r><w:rPr><w:rFonts w:ascii=\"Times New Roman\" w:hAnsi=\"Times New Roman\" "
        'w:eastAsia="SimSun"/><w:color w:val="000000"/></w:rPr>'
        f"<w:t>{safe_text}</w:t></w:r>"
        "</w:p>"
    )


def code_block(text: str) -> str:
    rows = []
    for line in text.splitlines() or [""]:
        rows.append(
            "<w:p>"
            '<w:pPr><w:pStyle w:val="Code"/></w:pPr>'
            '<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="SimSun"/>'
            '<w:color w:val="000000"/></w:rPr>'
            f"<w:t xml:space=\"preserve\">{html.escape(line)}</w:t></w:r>"
            "</w:p>"
        )
    return "".join(rows)


def table(table_lines: list[str]) -> str:
    if len(table_lines) < 2:
        return ""
    data_lines = [table_lines[0], *table_lines[2:]]
    rows = []
    for line in data_lines:
        cells = [clean_inline(cell.strip()) for cell in line.strip("|").split("|")]
        cell_xml = "".join(
            "<w:tc>"
            '<w:tcPr><w:tcW w:w="2400" w:type="dxa"/></w:tcPr>'
            f"{paragraph(cell)}"
            "</w:tc>"
            for cell in cells
        )
        rows.append(f"<w:tr>{cell_xml}</w:tr>")
    return (
        "<w:tbl>"
        "<w:tblPr>"
        '<w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        "</w:tblBorders>"
        "</w:tblPr>"
        f"{''.join(rows)}"
        "</w:tbl>"
    )


def styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{NS_W}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"/>
        <w:color w:val="000000"/>
        <w:sz w:val="21"/>
      </w:rPr>
    </w:rPrDefault>
  </w:docDefaults>
  {style('Normal', 'Normal', 21, False)}
  {style('Heading1', 'Heading 1', 32, True)}
  {style('Heading2', 'Heading 2', 28, True)}
  {style('Heading3', 'Heading 3', 24, True)}
  {style('ListParagraph', 'List Paragraph', 21, False, indent=True)}
  {style('Code', 'Code', 18, False, code=True)}
</w:styles>"""


def style(
    style_id: str,
    name: str,
    size: int,
    bold: bool,
    indent: bool = False,
    code: bool = False,
) -> str:
    rfonts = 'w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="SimSun"' if code else (
        'w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"'
    )
    bold_xml = "<w:b/>" if bold else ""
    indent_xml = '<w:pPr><w:ind w:left="420" w:hanging="180"/></w:pPr>' if indent else ""
    spacing_xml = "" if indent else '<w:pPr><w:spacing w:after="120"/></w:pPr>'
    return (
        f'<w:style w:type="paragraph" w:styleId="{style_id}">'
        f'<w:name w:val="{name}"/>'
        f"{indent_xml or spacing_xml}"
        "<w:rPr>"
        f"<w:rFonts {rfonts}/>"
        f"{bold_xml}"
        '<w:color w:val="000000"/>'
        f'<w:sz w:val="{size}"/>'
        "</w:rPr>"
        "</w:style>"
    )


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
</Types>"""


def root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"></Relationships>"""


def settings_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{NS_W}">
  <w:defaultTabStop w:val="420"/>
</w:settings>"""


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Create an XLSX companion spreadsheet with embedded thumbnails.

This avoids adding openpyxl/xlsxwriter dependencies by writing the small subset of
Office Open XML needed for one worksheet with images.
"""

from __future__ import annotations

import csv
import html
from io import BytesIO
from pathlib import Path
from typing import Dict, List
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "field_data" / "blue_cooler_may29_estimated_labels.csv"
IMAGE_DIR = ROOT / "data" / "raw" / "Green Crab AI 2026"
OUT_PATH = ROOT / "field_data" / "blue_cooler_may29_estimated_labels_with_thumbnails.xlsx"
THUMB_SIZE = 96


def xml(text: object) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def col_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def read_rows() -> List[Dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_thumbnail_bytes(image_path: Path) -> bytes:
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((THUMB_SIZE, THUMB_SIZE))
        canvas = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), "white")
        x = (THUMB_SIZE - image.width) // 2
        y = (THUMB_SIZE - image.height) // 2
        canvas.paste(image, (x, y))
        output = BytesIO()
        canvas.save(output, format="JPEG", quality=82)
        return output.getvalue()


def worksheet_xml(rows: List[Dict[str, str]], headers: List[str]) -> str:
    cols = ['<col min="1" max="1" width="15" customWidth="1"/>']
    for idx in range(2, len(headers) + 2):
        width = 18
        if headers[idx - 2] in {"notes", "view_angle_model", "sex_label_source"}:
            width = 38
        cols.append(f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>')

    xml_rows = ['<row r="1" ht="22" customHeight="1">']
    xml_rows[0] += '<c r="A1" t="inlineStr"><is><t>thumbnail</t></is></c>'
    for col_idx, header in enumerate(headers, start=1):
        cell = f"{col_name(col_idx)}1"
        xml_rows[0] += f'<c r="{cell}" t="inlineStr"><is><t>{xml(header)}</t></is></c>'
    xml_rows[0] += "</row>"

    for row_idx, row in enumerate(rows, start=2):
        parts = [f'<row r="{row_idx}" ht="78" customHeight="1">']
        for col_idx, header in enumerate(headers, start=1):
            cell = f"{col_name(col_idx)}{row_idx}"
            parts.append(f'<c r="{cell}" t="inlineStr"><is><t>{xml(row.get(header, ""))}</t></is></c>')
        parts.append("</row>")
        xml_rows.append("".join(parts))

    dimension = f"A1:{col_name(len(headers))}{len(rows) + 1}"
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="{dimension}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/><selection pane="bottomLeft"/></sheetView></sheetViews>
  <cols>{''.join(cols)}</cols>
  <sheetData>{''.join(xml_rows)}</sheetData>
  <drawing r:id="rId1"/>
</worksheet>'''


def drawing_xml(rows: List[Dict[str, str]]) -> str:
    anchors = []
    emu = 914400
    thumb_emu = int(0.92 * emu)
    for idx, _row in enumerate(rows, start=1):
        # Images occupy column A, row idx+1. Use oneCellAnchor with fixed ext.
        anchors.append(
            f'''<xdr:oneCellAnchor>
  <xdr:from><xdr:col>0</xdr:col><xdr:colOff>45720</xdr:colOff><xdr:row>{idx}</xdr:row><xdr:rowOff>45720</xdr:rowOff></xdr:from>
  <xdr:ext cx="{thumb_emu}" cy="{thumb_emu}"/>
  <xdr:pic>
    <xdr:nvPicPr><xdr:cNvPr id="{idx}" name="thumb_{idx}.jpg"/><xdr:cNvPicPr/></xdr:nvPicPr>
    <xdr:blipFill><a:blip r:embed="rId{idx}"/><a:stretch><a:fillRect/></a:stretch></xdr:blipFill>
    <xdr:spPr><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></xdr:spPr>
  </xdr:pic>
  <xdr:clientData/>
</xdr:oneCellAnchor>'''
        )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
{''.join(anchors)}
</xdr:wsDr>'''


def drawing_rels_xml(rows: List[Dict[str, str]]) -> str:
    rels = []
    for idx, _row in enumerate(rows, start=1):
        rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image{idx}.jpg"/>'
        )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{''.join(rels)}
</Relationships>'''


def build_xlsx() -> None:
    rows = read_rows()
    if not rows:
        raise ValueError(f"No rows found in {CSV_PATH}")
    headers = list(rows[0].keys())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUT_PATH, "w", ZIP_DEFLATED) as xlsx:
        xlsx.writestr(
            "[Content_Types].xml",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="jpg" ContentType="image/jpeg"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/drawings/drawing1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>''',
        )
        xlsx.writestr(
            "_rels/.rels",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>''',
        )
        xlsx.writestr(
            "xl/workbook.xml",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="blue_cooler_may29" sheetId="1" r:id="rId1"/></sheets>
</workbook>''',
        )
        xlsx.writestr(
            "xl/_rels/workbook.xml.rels",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>''',
        )
        xlsx.writestr(
            "xl/styles.xml",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
</styleSheet>''',
        )
        xlsx.writestr("xl/worksheets/sheet1.xml", worksheet_xml(rows, headers))
        xlsx.writestr(
            "xl/worksheets/_rels/sheet1.xml.rels",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/drawing1.xml"/>
</Relationships>''',
        )
        xlsx.writestr("xl/drawings/drawing1.xml", drawing_xml(rows))
        xlsx.writestr("xl/drawings/_rels/drawing1.xml.rels", drawing_rels_xml(rows))

        for idx, row in enumerate(rows, start=1):
            image_path = IMAGE_DIR / row["image_filename"]
            if image_path.exists():
                image_bytes = make_thumbnail_bytes(image_path)
            else:
                image_bytes = b""
            xlsx.writestr(f"xl/media/image{idx}.jpg", image_bytes)

    print(OUT_PATH)


if __name__ == "__main__":
    build_xlsx()

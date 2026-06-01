#!/usr/bin/env python3
"""
Create an XLSX review workbook from SAM3 bbox proposals.

The workbook embeds an overlay thumbnail and crop thumbnail per candidate, with
editable review_status/review_notes columns. Save a copy as CSV after review and
feed it to export_reviewed_bboxes_to_yolo.py.
"""

from __future__ import annotations

import argparse
import csv
import html
from io import BytesIO
from pathlib import Path
from typing import Dict, List
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image


THUMB_SIZE = 128
IMAGE_COLUMNS = ["overlay_thumbnail", "crop_thumbnail"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def xml(text: object) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def col_name(index: int) -> str:
    index += 1
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def thumbnail_bytes(path: Path) -> bytes:
    if not path.exists():
        return b""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.thumbnail((THUMB_SIZE, THUMB_SIZE))
        canvas = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), "white")
        x = (THUMB_SIZE - image.width) // 2
        y = (THUMB_SIZE - image.height) // 2
        canvas.paste(image, (x, y))
        out = BytesIO()
        canvas.save(out, format="JPEG", quality=84)
        return out.getvalue()


def worksheet_xml(rows: List[Dict[str, str]], headers: List[str]) -> str:
    all_headers = IMAGE_COLUMNS + headers
    cols = []
    for idx, header in enumerate(all_headers, start=1):
        width = 18 if header not in {"image_path", "overlay_path", "crop_path", "review_notes"} else 36
        cols.append(f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>')

    xml_rows = ['<row r="1" ht="24" customHeight="1">']
    for col_idx, header in enumerate(all_headers):
        cell = f"{col_name(col_idx)}1"
        xml_rows[0] += f'<c r="{cell}" t="inlineStr"><is><t>{xml(header)}</t></is></c>'
    xml_rows[0] += "</row>"

    for row_idx, row in enumerate(rows, start=2):
        parts = [f'<row r="{row_idx}" ht="102" customHeight="1">']
        for col_idx, header in enumerate(headers, start=len(IMAGE_COLUMNS)):
            cell = f"{col_name(col_idx)}{row_idx}"
            parts.append(f'<c r="{cell}" t="inlineStr"><is><t>{xml(row.get(header, ""))}</t></is></c>')
        parts.append("</row>")
        xml_rows.append("".join(parts))

    dimension = f"A1:{col_name(len(all_headers) - 1)}{len(rows) + 1}"
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="{dimension}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/><selection pane="bottomLeft"/></sheetView></sheetViews>
  <cols>{''.join(cols)}</cols>
  <sheetData>{''.join(xml_rows)}</sheetData>
  <drawing r:id="rId1"/>
</worksheet>'''


def drawing_xml(num_images: int) -> str:
    anchors = []
    emu = 914400
    thumb_emu = int(1.05 * emu)
    rel_id = 1
    for row_idx in range(1, num_images + 1):
        for col_idx in range(len(IMAGE_COLUMNS)):
            anchors.append(
                f'''<xdr:oneCellAnchor>
  <xdr:from><xdr:col>{col_idx}</xdr:col><xdr:colOff>45720</xdr:colOff><xdr:row>{row_idx}</xdr:row><xdr:rowOff>45720</xdr:rowOff></xdr:from>
  <xdr:ext cx="{thumb_emu}" cy="{thumb_emu}"/>
  <xdr:pic>
    <xdr:nvPicPr><xdr:cNvPr id="{rel_id}" name="thumb_{rel_id}.jpg"/><xdr:cNvPicPr/></xdr:nvPicPr>
    <xdr:blipFill><a:blip r:embed="rId{rel_id}"/><a:stretch><a:fillRect/></a:stretch></xdr:blipFill>
    <xdr:spPr><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></xdr:spPr>
  </xdr:pic>
  <xdr:clientData/>
</xdr:oneCellAnchor>'''
            )
            rel_id += 1
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
{''.join(anchors)}
</xdr:wsDr>'''


def drawing_rels_xml(num_media: int) -> str:
    rels = [
        f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image{idx}.jpg"/>'
        for idx in range(1, num_media + 1)
    ]
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>'''


def build_xlsx(proposals_path: Path, out_path: Path) -> None:
    rows = read_rows(proposals_path)
    if not rows:
        raise ValueError(f"No rows found in {proposals_path}")
    headers = list(rows[0].keys())
    base_dir = proposals_path.parent
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out_path, "w", ZIP_DEFLATED) as xlsx:
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
  <sheets><sheet name="bbox_review" sheetId="1" r:id="rId1"/></sheets>
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
        xlsx.writestr("xl/styles.xml", '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>')
        xlsx.writestr("xl/worksheets/sheet1.xml", worksheet_xml(rows, headers))
        xlsx.writestr(
            "xl/worksheets/_rels/sheet1.xml.rels",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/drawing1.xml"/>
</Relationships>''',
        )
        media_count = len(rows) * len(IMAGE_COLUMNS)
        xlsx.writestr("xl/drawings/drawing1.xml", drawing_xml(len(rows)))
        xlsx.writestr("xl/drawings/_rels/drawing1.xml.rels", drawing_rels_xml(media_count))
        media_id = 1
        for row in rows:
            for field in ("overlay_path", "crop_path"):
                xlsx.writestr(f"xl/media/image{media_id}.jpg", thumbnail_bytes(base_dir / row[field]))
                media_id += 1


def main() -> None:
    args = parse_args()
    build_xlsx(args.proposals, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

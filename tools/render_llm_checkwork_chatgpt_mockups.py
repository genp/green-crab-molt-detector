#!/usr/bin/env python3
from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from matplotlib import font_manager


W = 1672
H = 941
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "presentation_templates"
OUT_DIR = ROOT / "presentation_mockups" / "llm_checkwork_chatgpt"

INK = (20, 33, 48, 255)
MUTED = (95, 107, 118, 255)
BLUE = (41, 93, 168, 255)
TEAL = (28, 125, 121, 255)
GREEN = (36, 131, 80, 255)
AMBER = (194, 124, 20, 255)
RED = (189, 62, 70, 255)
LINE = (212, 221, 230, 255)
SOFT = (243, 247, 250, 255)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    prop = font_manager.FontProperties(
        family="DejaVu Sans",
        weight="bold" if bold else "normal",
    )
    path = font_manager.findfont(prop)
    return ImageFont.truetype(path, size=size)


def bg(name: str) -> Image.Image:
    return Image.open(TEMPLATE_DIR / name).convert("RGBA")


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.ImageFont, max_w: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current = [words[0]]
        for word in words[1:]:
            trial = " ".join(current + [word])
            if draw.textlength(trial, font=f) <= max_w:
                current.append(word)
            else:
                lines.append(" ".join(current))
                current = [word]
        lines.append(" ".join(current))
    return lines


def write_wrapped(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    f: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    line_gap: int = 6,
    align: str = "left",
) -> int:
    x0, y0, x1, _ = box
    max_w = x1 - x0
    lines = wrap(draw, text, f, max_w)
    line_h = draw.textbbox((0, 0), "Ag", font=f)[3] + line_gap
    y = y0
    for line in lines:
        if align == "center":
            w = draw.textlength(line, font=f)
            x = x0 + (max_w - w) / 2
        elif align == "right":
            w = draw.textlength(line, font=f)
            x = x1 - w
        else:
            x = x0
        draw.text((x, y), line, font=f, fill=fill)
        y += line_h
    return y


def fit_font(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], start: int, end: int, bold: bool = False) -> ImageFont.ImageFont:
    x0, y0, x1, y1 = box
    max_w = x1 - x0
    max_h = y1 - y0
    for size in range(start, end - 1, -1):
        f = font(size, bold=bold)
        lines = wrap(draw, text, f, max_w)
        line_h = draw.textbbox((0, 0), "Ag", font=f)[3] + 6
        if len(lines) * line_h - 6 <= max_h and all(draw.textlength(line, font=f) <= max_w for line in lines):
            return f
    return font(end, bold=bold)


def rounded(
    img: Image.Image,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] = LINE,
    width: int = 2,
    radius: int = 16,
) -> None:
    ImageDraw.Draw(img).rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def shadow(img: Image.Image, box: tuple[int, int, int, int], radius: int = 16) -> None:
    tmp = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    d.rounded_rectangle(box, radius=radius, fill=(0, 0, 0, 38))
    tmp = tmp.filter(ImageFilter.GaussianBlur(16))
    img.alpha_composite(tmp)


def frame_image(base: Image.Image, img: Image.Image, box: tuple[int, int, int, int], radius: int = 14) -> None:
    shadow(base, box, radius=radius)
    rounded(base, box, (255, 255, 255, 255), outline=LINE, width=2, radius=radius)
    inner = (box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2)
    fitted = ImageOps.contain(img.convert("RGBA"), (inner[2] - inner[0], inner[3] - inner[1]))
    pad = Image.new("RGBA", (inner[2] - inner[0], inner[3] - inner[1]), (255, 255, 255, 255))
    pad.alpha_composite(fitted, ((pad.width - fitted.width) // 2, (pad.height - fitted.height) // 2))
    base.alpha_composite(pad, (inner[0], inner[1]))


def pill(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int, int], text: str) -> None:
    draw.rounded_rectangle(box, radius=999, fill=fill)
    f = font(18, bold=True)
    w = draw.textlength(text, font=f)
    h = draw.textbbox((0, 0), "Ag", font=f)[3]
    x = box[0] + (box[2] - box[0] - w) / 2
    y = box[1] + (box[3] - box[1] - h) / 2 - 1
    draw.text((x, y), text, font=f, fill=(255, 255, 255, 255))


def title_block(img: Image.Image, eyebrow: str, title: str, panel: tuple[int, int, int, int] = (930, 104, 1600, 860)) -> int:
    d = ImageDraw.Draw(img)
    x0, y0, x1, _ = panel
    d.text((x0, y0), eyebrow.upper(), font=font(15, bold=True), fill=MUTED)
    tf = fit_font(d, title, (x0, y0 + 20, x1, y0 + 150), 40, 28, bold=True)
    return write_wrapped(d, (x0, y0 + 24, x1, y0 + 170), title, tf, INK, line_gap=6)


def callout(img: Image.Image, box: tuple[int, int, int, int], title: str, body: str, accent: tuple[int, int, int, int]) -> None:
    d = ImageDraw.Draw(img)
    rounded(img, box, (255, 255, 255, 255), outline=LINE, width=2, radius=16)
    d.rectangle((box[0], box[1], box[0] + 6, box[3]), fill=accent)
    d.text((box[0] + 18, box[1] + 18), title, font=font(20, bold=True), fill=INK)
    write_wrapped(d, (box[0] + 18, box[1] + 56, box[2] - 18, box[3] - 18), body, font(18), MUTED, line_gap=5)


def codebox(img: Image.Image, box: tuple[int, int, int, int], text: str) -> None:
    d = ImageDraw.Draw(img)
    rounded(img, box, (18, 26, 36, 255), outline=(18, 26, 36, 255), width=1, radius=16)
    f = font(16)
    y = box[1] + 18
    max_w = box[2] - box[0] - 40
    line_h = d.textbbox((0, 0), "Ag", font=f)[3] + 8
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            y += line_h // 2
            continue
        for line in wrap(d, paragraph, f, max_w):
            d.text((box[0] + 20, y), line, font=f, fill=(230, 236, 243, 255))
            y += line_h


def title_slide() -> Image.Image:
    img = bg("ccai_agentic_title_no_footer.png")
    d = ImageDraw.Draw(img)
    x0, x1 = 942, 1600
    pill(d, (x0, 120, x0 + 260, 160), BLUE, "ChatGPT web workflow")
    title = "Using ChatGPT Web to explain `llm_checkwork.py`"
    tf = fit_font(d, title, (x0, 184, x1, 360), 42, 30, bold=True)
    write_wrapped(d, (x0, 192, x1, 360), title, tf, INK, line_gap=7)
    sf = fit_font(d, "The source data lives in the course materials folder on disk.", (x0, 372, x1, 430), 24, 18)
    write_wrapped(d, (x0, 378, x1, 430), "The source data lives in the course materials folder on disk.", sf, MUTED, line_gap=5)
    callout(
        img,
        (x0, 470, x1, 680),
        "Source locations",
        "Notebook: `~/Downloads/llm_checkwork.py`\nData: `~/Practical ML Tools for Climate Change Course Materials/`",
        TEAL,
    )
    d.text((x0, 720), "Speaker notes deck mockup", font=font(18, bold=True), fill=BLUE)
    write_wrapped(
        d,
        (x0, 754, x1, 826),
        "Use the web version of ChatGPT to generate a clean step-by-step explanation, then verify it against the local files.",
        font(15),
        MUTED,
        line_gap=4,
    )
    return img


def prompt_slide() -> Image.Image:
    img = bg("ccai_agentic_content_no_footer.png")
    d = ImageDraw.Draw(img)
    title_block(img, "Prompt pattern", "Ask for structure first, not code first")
    code = (
        "Read `~/Downloads/llm_checkwork.py` and explain each block in order.\n"
        "Use the source data in `~/Practical ML Tools for Climate Change Course Materials/`:\n"
        "  - `icecore/vostok.icecore.co2`\n"
        "  - `icecore/vostok.1999.temp.dat`\n"
        "  - `modern_climate/co2_mm_mlo.txt`\n"
        "  - `modern_climate/Land_and_Ocean_complete.txt`\n\n"
        "Return:\n"
        "  1. a concise step list\n"
        "  2. file dependencies\n"
        "  3. assumptions that need verification"
    )
    codebox(img, (930, 250, 1600, 720), code)
    callout(
        img,
        (930, 752, 1600, 842),
        "Why this works",
        "It keeps ChatGPT focused on explanation and provenance before it starts narrating results.",
        BLUE,
    )
    return img


def notebook_map_slide() -> Image.Image:
    img = bg("ccai_agentic_content_footer.png")
    d = ImageDraw.Draw(img)
    title_block(img, "Notebook map", "What the script actually does")
    steps = [
        "Load Vostok CO2 and temperature tables from the course materials.",
        "Merge them on age using nearest-time alignment.",
        "Sort in reverse chronology and plot temperature and CO2.",
        "Compute Pearson correlation and mutual information.",
        "Fit orbital-cycle priors with 2-cycle and 3-cycle models.",
        "Compare paleo and modern CO2 with a Welch t-test.",
        "Forecast temperature anomalies with linear, bootstrap, and Bayesian models.",
    ]
    y = 252
    for i, step in enumerate(steps, start=1):
        pill(d, (930, y, 980, y + 34), BLUE if i < 4 else TEAL, str(i))
        write_wrapped(d, (1000, y - 2, 1600, y + 70), step, font(20), INK, line_gap=4)
        y += 74
    callout(
        img,
        (930, 792, 1600, 858),
        "Takeaway",
        "The deck is about supervision: use ChatGPT to map the workflow, then verify every step locally.",
        GREEN,
    )
    return img


def file_map_slide() -> Image.Image:
    img = bg("ccai_agentic_figure_no_footer.png")
    d = ImageDraw.Draw(img)
    title_block(img, "Source data", "Make ChatGPT name the files, not just the concepts")
    left = (930, 244, 1260, 490)
    right = (1294, 244, 1600, 490)
    for box, title, accent, items in [
        (
            left,
            "Ice core",
            BLUE,
            [
                "`icecore/`\n`vostok.icecore.co2`",
                "`icecore/`\n`vostok.1999.temp.dat`",
            ],
        ),
        (
            right,
            "Modern climate",
            TEAL,
            [
                "`modern_climate/`\n`co2_mm_mlo.txt`",
                "`modern_climate/`\n`Land_and_Ocean_complete.txt`",
            ],
        ),
    ]:
        rounded(img, box, (255, 255, 255, 255), outline=accent, width=3, radius=18)
        d.rectangle((box[0], box[1], box[2], box[1] + 50), fill=accent)
        d.text((box[0] + 18, box[1] + 15), title, font=font(20, bold=True), fill=(255, 255, 255, 255))
        yy = box[1] + 82
        for item in items:
            rounded(img, (box[0] + 16, yy, box[2] - 16, yy + 78), (248, 250, 252, 255), outline=(221, 229, 236, 255), width=1, radius=12)
            write_wrapped(d, (box[0] + 28, yy + 12, box[2] - 28, yy + 68), item, font(13, bold=True), INK, line_gap=1)
            yy += 90
    callout(
        img,
        (930, 540, 1600, 842),
        "Related context",
        "All files sit under `~/Practical ML Tools for Climate Change Course Materials/`. Keep the explanation tied to the file map; if the model cannot point to the exact files, ask it to redo the summary.",
        AMBER,
    )
    return img


def verification_slide() -> Image.Image:
    img = bg("ccai_agentic_section_footer.png")
    d = ImageDraw.Draw(img)
    title_block(img, "Verification", "Use ChatGPT as a checker, not as the source of truth")
    bullets = [
        "Confirm `skiprows` and column names match the files.",
        "Confirm the merge direction and age alignment are intentional.",
        "Check units: ppmv, degrees C, and years before present.",
        "Verify the forecast horizon and model assumptions locally.",
        "Re-run the notebook blocks that use curve fitting or PyMC.",
    ]
    y = 255
    for b in bullets:
        d.ellipse((936, y + 4, 950, y + 18), fill=BLUE)
        write_wrapped(d, (964, y, 1600, y + 66), b, font(20), INK, line_gap=4)
        y += 68
    callout(
        img,
        (930, 620, 1600, 842),
        "If something diverges",
        "Fix the prompt and ask again before trusting the summary. The local files remain the authority.",
        RED,
    )
    return img


def workflow_slide() -> Image.Image:
    img = bg("ccai_agentic_title_no_footer.png")
    d = ImageDraw.Draw(img)
    x0, x1 = 942, 1600
    d.text((x0, 120), "Workflow", font=font(15, bold=True), fill=MUTED)
    tf = fit_font(d, "A practical loop: ask, verify, revise", (x0, 148, x1, 240), 40, 30, bold=True)
    write_wrapped(d, (x0, 152, x1, 250), "A practical loop: ask, verify, revise", tf, INK, line_gap=6)
    steps = [
        "Paste the notebook export.",
        "Ask ChatGPT for a block-by-block summary.",
        "Ask for the file map and the assumptions.",
        "Check the source files locally.",
        "Revise until the explanation matches the code.",
        "Turn the verified steps into speaker notes or slides.",
    ]
    y = 320
    for i, step in enumerate(steps, start=1):
        pill(d, (x0, y, x0 + 54, y + 34), BLUE if i < 4 else TEAL, str(i))
        write_wrapped(d, (x0 + 74, y - 1, x1, y + 40), step, font(19), INK, line_gap=4)
        y += 58
    callout(
        img,
        (x0, 680, x1, 842),
        "Bottom line",
        "The model is good at explanation and organization. The local files are still the authority.",
        GREEN,
    )
    return img


def save(path: Path, img: Image.Image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def contact_sheet(paths: list[Path], out_path: Path) -> Path:
    thumbs = []
    for p in paths:
        im = Image.open(p).convert("RGBA")
        thumb = ImageOps.contain(im, (360, 202))
        canvas = Image.new("RGBA", (380, 236), (255, 255, 255, 255))
        canvas.alpha_composite(thumb, ((380 - thumb.width) // 2, 8))
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle((8, 8, 372, 228), radius=14, outline=(214, 221, 229, 255), width=2)
        d.text((16, 208), p.stem, font=font(14), fill=MUTED)
        thumbs.append(canvas)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGBA", (cols * 380 + 24, rows * 236 + 24), (247, 249, 250, 255))
    for i, thumb in enumerate(thumbs):
        x = 12 + (i % cols) * 380
        y = 12 + (i // cols) * 236
        sheet.alpha_composite(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def main() -> None:
    slides = [
        ("01_title.png", title_slide()),
        ("02_prompt_pattern.png", prompt_slide()),
        ("03_notebook_map.png", notebook_map_slide()),
        ("04_file_map.png", file_map_slide()),
        ("05_verification.png", verification_slide()),
        ("06_workflow_loop.png", workflow_slide()),
    ]
    paths = [save(OUT_DIR / name, img) for name, img in slides]
    contact_sheet(paths, OUT_DIR / "llm_checkwork_chatgpt_contact_sheet.png")
    print(f"Wrote {len(paths)} slide mockups to {OUT_DIR}")


if __name__ == "__main__":
    main()

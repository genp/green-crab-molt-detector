#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from matplotlib import font_manager


W = 1672
H = 941
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "presentation_templates"
OUT_DIR = ROOT / "presentation_mockups" / "llm_checkwork_chatgpt"

INK = (24, 35, 46, 255)
MUTED = (94, 107, 120, 255)
TITLE = (16, 91, 105, 255)
TEAL = (37, 131, 137, 255)
BLUE = (37, 101, 173, 255)
GREEN = (36, 127, 82, 255)
AMBER = (207, 128, 12, 255)
RED = (214, 53, 53, 255)
LINE = (172, 208, 220, 255)
LIGHT = (246, 249, 251, 255)
FOOTER = (15, 94, 131, 255)
FOOTER_DARK = (7, 70, 118, 255)


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont:
    family = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    prop = font_manager.FontProperties(family=family, weight="bold" if bold else "normal")
    return ImageFont.truetype(font_manager.findfont(prop), size=size)


def make_bg() -> Image.Image:
    return Image.new("RGBA", (W, H), (255, 255, 255, 255))


def add_waves(img: Image.Image) -> None:
    d = ImageDraw.Draw(img)
    for i in range(5):
        y = 72 + i * 6
        d.arc((20, y, 340, y + 34), 180, 360, fill=(205, 227, 233, 150), width=1)
        d.arc((1334, y, 1652, y + 34), 180, 360, fill=(205, 227, 233, 150), width=1)
    d.line((312, 78, 408, 78), fill=(95, 175, 186, 255), width=2)
    d.line((1262, 78, 1354, 78), fill=(95, 175, 186, 255), width=2)
    for x in [295, 408, 1245, 1354]:
        d.ellipse((x - 3, 75, x + 3, 81), fill=(95, 175, 186, 255))


def add_footer_band(img: Image.Image, text: str) -> None:
    d = ImageDraw.Draw(img)
    band = Image.new("RGBA", (W, 94), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    for y in range(94):
        t = y / 93
        r = int(FOOTER[0] * (1 - t) + FOOTER_DARK[0] * t)
        g = int(FOOTER[1] * (1 - t) + FOOTER_DARK[1] * t)
        b = int(FOOTER[2] * (1 - t) + FOOTER_DARK[2] * t)
        bd.line((0, y, W, y), fill=(r, g, b, 255))
    img.alpha_composite(band, (0, H - 94))
    d.line((0, H - 94, W, H - 94), fill=(238, 189, 76, 255), width=2)
    d.ellipse((78, H - 74, 128, H - 24), outline=(255, 255, 255, 255), width=2)
    d.text((93, H - 61), "?", font=font(22, bold=True), fill=(255, 255, 255, 255), anchor="mm")
    d.text((154, H - 56), text, font=font(23, bold=True), fill=(255, 255, 255, 255))


def rounded(img: Image.Image, box: tuple[int, int, int, int], fill, outline=LINE, width=2, radius=18) -> None:
    ImageDraw.Draw(img).rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def shadow(img: Image.Image, box: tuple[int, int, int, int], radius: int = 18) -> None:
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle(box, radius=radius, fill=(0, 0, 0, 38))
    sh = sh.filter(ImageFilter.GaussianBlur(16))
    img.alpha_composite(sh)


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.ImageFont, max_w: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
            continue
        words = para.split()
        cur = [words[0]]
        for word in words[1:]:
            trial = " ".join(cur + [word])
            if draw.textlength(trial, font=f) <= max_w:
                cur.append(word)
            else:
                lines.append(" ".join(cur))
                cur = [word]
        lines.append(" ".join(cur))
    return lines


def write(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, f: ImageFont.ImageFont, fill, line_gap: int = 6, align: str = "left") -> int:
    x0, y0, x1, _ = box
    lines = wrap(draw, text, f, x1 - x0)
    lh = draw.textbbox((0, 0), "Ag", font=f)[3] + line_gap
    y = y0
    for line in lines:
        if align == "center":
            w = draw.textlength(line, font=f)
            x = x0 + ((x1 - x0) - w) / 2
        elif align == "right":
            w = draw.textlength(line, font=f)
            x = x1 - w
        else:
            x = x0
        draw.text((x, y), line, font=f, fill=fill)
        y += lh
    return y


def title_line(draw: ImageDraw.ImageDraw, title: str, y: int = 24) -> int:
    f = font(48, bold=True)
    return write(draw, (160, y, W - 160, y + 82), title, f, TITLE, line_gap=2, align="center")


def decorate_template(img: Image.Image, variant: str) -> None:
    add_waves(img)
    d = ImageDraw.Draw(img)
    if variant == "title":
        for x in [90, 144, 1528, 1584]:
            d.line((x, 118, x + 120 if x < 800 else x - 120, 118), fill=(167, 219, 228, 120), width=1)
        d.rounded_rectangle((72, 148, 1600, 790), radius=22, outline=(223, 235, 241, 255), width=1)
    elif variant == "content":
        for x, y, w, h in [(72, 150, 700, 250), (900, 150, 700, 250), (72, 430, 700, 250), (900, 430, 700, 250)]:
            d.rounded_rectangle((x, y, x + w, y + h), radius=18, outline=(228, 236, 241, 255), width=1)
    elif variant == "section":
        d.rounded_rectangle((112, 160, 1560, 800), radius=24, outline=(223, 235, 241, 255), width=1)
        d.line((280, 430, 1392, 430), fill=(223, 235, 241, 255), width=2)


def write_template(path: Path, variant: str) -> None:
    img = make_bg()
    decorate_template(img, variant)
    add_footer_band(img, "Use ChatGPT to generate the steps, then verify against the local files.")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def ensure_templates() -> None:
    write_template(TEMPLATE_DIR / "llm_checkwork_title_template.png", "title")
    write_template(TEMPLATE_DIR / "llm_checkwork_content_template.png", "content")
    write_template(TEMPLATE_DIR / "llm_checkwork_section_template.png", "section")


def load(name: str) -> Image.Image:
    return Image.open(TEMPLATE_DIR / name).convert("RGBA")


def card(img: Image.Image, box: tuple[int, int, int, int], title: str, body: str, accent: tuple[int, int, int, int], body_size: int = 20) -> None:
    d = ImageDraw.Draw(img)
    shadow(img, box)
    rounded(img, box, (255, 255, 255, 255), outline=LINE, width=2, radius=18)
    d.rectangle((box[0], box[1], box[0] + 6, box[3]), fill=accent)
    d.text((box[0] + 18, box[1] + 16), title, font=font(22, bold=True), fill=INK)
    write(d, (box[0] + 18, box[1] + 56, box[2] - 18, box[3] - 18), body, font(body_size), MUTED, line_gap=5)


def place_image(img: Image.Image, src: Path, box: tuple[int, int, int, int], radius: int = 16) -> None:
    shadow(img, box, radius=radius)
    rounded(img, box, (255, 255, 255, 255), outline=LINE, width=2, radius=radius)
    inner = (box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2)
    fitted = ImageOps.contain(Image.open(src).convert("RGBA"), (inner[2] - inner[0], inner[3] - inner[1]))
    paste_x = inner[0] + ((inner[2] - inner[0]) - fitted.width) // 2
    paste_y = inner[1] + ((inner[3] - inner[1]) - fitted.height) // 2
    img.alpha_composite(fitted, (paste_x, paste_y))


def pill(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int, int], text: str) -> None:
    draw.rounded_rectangle(box, radius=999, fill=fill)
    f = font(18, bold=True)
    w = draw.textlength(text, font=f)
    h = draw.textbbox((0, 0), "Ag", font=f)[3]
    draw.text((box[0] + (box[2] - box[0] - w) / 2, box[1] + (box[3] - box[1] - h) / 2 - 1), text, font=f, fill=(255, 255, 255, 255))


def title_slide() -> Image.Image:
    img = load("llm_checkwork_title_template.png")
    d = ImageDraw.Draw(img)
    title_line(d, "Using ChatGPT Web to explain `llm_checkwork.py`")
    d.text((160, 100), "CHATGPT WEB WORKFLOW", font=font(16, bold=True), fill=MUTED)
    card(
        img,
        (92, 185, 780, 610),
        "Source locations",
        "Notebook: `~/Downloads/llm_checkwork.py`\nData: `~/Practical ML Tools for Climate Change Course Materials/`",
        BLUE,
        body_size=19,
    )
    card(
        img,
        (892, 185, 1580, 610),
        "How to use ChatGPT web",
        "Ask for the block structure first.\nAsk for the file map next.\nThen verify every claim locally before you trust the summary.",
        TEAL,
        body_size=19,
    )
    card(
        img,
        (92, 652, 1580, 790),
        "What this deck is for",
        "Use the web version of ChatGPT to generate a clean step-by-step explanation and a file provenance map, not a final source of truth.",
        GREEN,
        body_size=19,
    )
    return img


def prompt_slide() -> Image.Image:
    img = load("llm_checkwork_content_template.png")
    d = ImageDraw.Draw(img)
    title_line(d, "Ask for structure first, not code first")
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
    card(img, (90, 180, 1100, 760), "Prompt to paste into ChatGPT web", code, BLUE, body_size=19)
    card(
        img,
        (1140, 180, 1590, 760),
        "Why this works",
        "It keeps ChatGPT focused on explanation and provenance before it starts narrating results.\n\nTreat the first answer as a draft, then rerun the prompt if the file map is vague.",
        RED,
        body_size=18,
    )
    return img


def notebook_map_slide() -> Image.Image:
    img = load("llm_checkwork_content_template.png")
    d = ImageDraw.Draw(img)
    title_line(d, "What the script actually does")
    plot = TEMPLATE_DIR / "llm_checkwork_temp_co2_plot.png"
    place_image(img, plot, (92, 184, 940, 606))
    steps = [
        "Load Vostok CO2 and temperature tables from the course materials.",
        "Merge them on age using nearest-time alignment.",
        "Sort in reverse chronology and plot temperature and CO2.",
        "Compute Pearson correlation and mutual information.",
        "Fit orbital-cycle priors with 2-cycle and 3-cycle models.",
        "Compare paleo and modern CO2 with a Welch t-test.",
        "Forecast temperature anomalies with linear, bootstrap, and Bayesian models.",
    ]
    card(img, (980, 184, 1580, 606), "Notebook map", "", GREEN, body_size=18)
    y = 248
    for i, step in enumerate(steps, start=1):
        pill(d, (1024, y, 1070, y + 34), BLUE if i <= 3 else TEAL, str(i))
        write(d, (1098, y - 1, 1540, y + 44), step, font(18), INK, line_gap=4)
        y += 50
    card(
        img,
        (92, 656, 1580, 790),
        "Takeaway",
        "The model is good at explanation and organization. The local files remain the authority.",
        AMBER,
        body_size=18,
    )
    return img


def file_map_slide() -> Image.Image:
    img = load("llm_checkwork_content_template.png")
    d = ImageDraw.Draw(img)
    title_line(d, "Make ChatGPT name the files, not just the concepts")
    card(
        img,
        (92, 184, 790, 560),
        "Ice core",
        "`icecore/`\n`vostok.icecore.co2`\n\n`icecore/`\n`vostok.1999.temp.dat`",
        BLUE,
        body_size=19,
    )
    card(
        img,
        (882, 184, 1580, 560),
        "Modern climate",
        "`modern_climate/`\n`co2_mm_mlo.txt`\n\n`modern_climate/`\n`Land_and_Ocean_complete.txt`",
        TEAL,
        body_size=19,
    )
    card(
        img,
        (92, 612, 1580, 790),
        "Related context",
        "All files sit under `~/Practical ML Tools for Climate Change Course Materials/`. Keep the explanation tied to the file map; if the model cannot point to the exact files, ask it to redo the summary.",
        AMBER,
        body_size=18,
    )
    return img


def verification_slide() -> Image.Image:
    img = load("llm_checkwork_content_template.png")
    d = ImageDraw.Draw(img)
    title_line(d, "Use ChatGPT as a checker, not as the source of truth")
    card(
        img,
        (92, 184, 1050, 790),
        "Verification checklist",
        "• Confirm `skiprows` and column names match the files.\n"
        "• Confirm the merge direction and age alignment are intentional.\n"
        "• Check units: ppmv, degrees C, and years before present.\n"
        "• Verify the forecast horizon and model assumptions locally.\n"
        "• Re-run the notebook blocks that use curve fitting or PyMC.",
        GREEN,
        body_size=19,
    )
    card(
        img,
        (1120, 184, 1580, 790),
        "If something diverges",
        "The answer can look polished and still be wrong.\n\nFix the prompt and ask again before trusting the summary.",
        RED,
        body_size=18,
    )
    return img


def workflow_slide() -> Image.Image:
    img = load("llm_checkwork_section_template.png")
    d = ImageDraw.Draw(img)
    title_line(d, "A practical loop: ask, verify, revise")
    steps = [
        "Paste the notebook export.",
        "Ask ChatGPT for a block-by-block summary.",
        "Ask for the file map and the assumptions.",
        "Check the source files locally.",
        "Revise until the explanation matches the code.",
        "Turn the verified steps into speaker notes or slides.",
    ]
    card(img, (112, 214, 1560, 690), "Workflow", "", TEAL, body_size=18)
    x = 160
    y = 286
    for i, step in enumerate(steps, start=1):
        pill(d, (x, y, x + 52, y + 34), BLUE if i <= 3 else GREEN, str(i))
        write(d, (x + 72, y - 1, 1480, y + 40), step, font(20), INK, line_gap=4)
        y += 58
    card(
        img,
        (112, 728, 1560, 790),
        "Bottom line",
        "The model is good at explanation and organization. The local files are still the authority.",
        AMBER,
        body_size=18,
    )
    return img


def save(path: Path, img: Image.Image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def contact_sheet(paths: list[Path], out: Path) -> None:
    thumbs = []
    for p in paths:
        im = Image.open(p).convert("RGBA")
        thumb = Image.new("RGBA", (380, 236), (255, 255, 255, 255))
        small = im.copy()
        small.thumbnail((360, 202))
        thumb.alpha_composite(small, ((380 - small.width) // 2, 10))
        d = ImageDraw.Draw(thumb)
        d.rounded_rectangle((8, 8, 372, 228), radius=16, outline=(214, 221, 229, 255), width=2)
        d.text((18, 208), p.stem, font=font(14), fill=MUTED)
        thumbs.append(thumb)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGBA", (cols * 380 + 24, rows * 236 + 24), LIGHT)
    for i, thumb in enumerate(thumbs):
        x = 12 + (i % cols) * 380
        y = 12 + (i // cols) * 236
        sheet.alpha_composite(thumb, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main() -> None:
    ensure_templates()
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
    print(f"Wrote {len(paths)} slides to {OUT_DIR}")


if __name__ == "__main__":
    main()

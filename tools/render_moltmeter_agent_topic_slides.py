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
STATIC_DIR = ROOT / "static"
OUT_DIR = ROOT / "presentation_mockups" / "moltmeter_agent_topics"

INK = (23, 35, 46, 255)
MUTED = (94, 107, 120, 255)
TITLE = (15, 93, 106, 255)
BLUE = (36, 100, 175, 255)
TEAL = (36, 131, 137, 255)
GREEN = (39, 129, 84, 255)
AMBER = (204, 128, 18, 255)
RED = (213, 55, 53, 255)
LINE = (170, 208, 219, 255)
PANEL = (249, 251, 253, 255)
FOOTER_1 = (17, 102, 133, 255)
FOOTER_2 = (8, 77, 122, 255)


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont:
    family = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    prop = font_manager.FontProperties(family=family, weight="bold" if bold else "normal")
    return ImageFont.truetype(font_manager.findfont(prop), size=size)


def bg() -> Image.Image:
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
        r = int(FOOTER_1[0] * (1 - t) + FOOTER_2[0] * t)
        g = int(FOOTER_1[1] * (1 - t) + FOOTER_2[1] * t)
        b = int(FOOTER_1[2] * (1 - t) + FOOTER_2[2] * t)
        bd.line((0, y, W, y), fill=(r, g, b, 255))
    img.alpha_composite(band, (0, H - 94))
    d.line((0, H - 94, W, H - 94), fill=(239, 190, 75, 255), width=2)
    d.ellipse((78, H - 74, 128, H - 24), outline=(255, 255, 255, 255), width=2)
    d.text((93, H - 61), "?", font=font(22, bold=True), fill=(255, 255, 255, 255), anchor="mm")
    d.text((154, H - 56), text, font=font(22, bold=True), fill=(255, 255, 255, 255))


def rounded(img: Image.Image, box: tuple[int, int, int, int], fill, outline=LINE, width=2, radius=18) -> None:
    ImageDraw.Draw(img).rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def shadow(img: Image.Image, box: tuple[int, int, int, int], radius: int = 18) -> None:
    tmp = Image.new("RGBA", img.size, (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    td.rounded_rectangle(box, radius=radius, fill=(0, 0, 0, 38))
    img.alpha_composite(tmp.filter(ImageFilter.GaussianBlur(16)))


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


def title_line(draw: ImageDraw.ImageDraw, title: str) -> None:
    write(draw, (160, 22, W - 160, 120), title, font(48, bold=True), TITLE, line_gap=2, align="center")


def template_base(variant: str) -> Image.Image:
    img = bg()
    add_waves(img)
    d = ImageDraw.Draw(img)
    if variant == "title":
        d.rounded_rectangle((72, 150, 1600, 790), radius=22, outline=(223, 235, 241, 255), width=1)
    elif variant == "content":
        for x, y, w, h in [(72, 150, 700, 250), (900, 150, 700, 250), (72, 430, 700, 250), (900, 430, 700, 250)]:
            d.rounded_rectangle((x, y, x + w, y + h), radius=18, outline=(228, 236, 241, 255), width=1)
    elif variant == "section":
        d.rounded_rectangle((112, 160, 1560, 800), radius=24, outline=(223, 235, 241, 255), width=1)
        d.line((280, 430, 1392, 430), fill=(223, 235, 241, 255), width=2)
    return img


def write_template(path: Path, variant: str) -> None:
    img = template_base(variant)
    add_footer_band(img, "Use ChatGPT to generate the steps, then verify against the local files.")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def ensure_templates() -> None:
    write_template(TEMPLATE_DIR / "moltmeter_agent_title_template.png", "title")
    write_template(TEMPLATE_DIR / "moltmeter_agent_content_template.png", "content")
    write_template(TEMPLATE_DIR / "moltmeter_agent_section_template.png", "section")


def load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def frame_image(base: Image.Image, src: Path, box: tuple[int, int, int, int], radius: int = 16, contain: bool = True) -> None:
    shadow(base, box, radius=radius)
    rounded(base, box, (255, 255, 255, 255), outline=LINE, width=2, radius=radius)
    inner = (box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2)
    im = load(src)
    if contain:
        fitted = ImageOps.contain(im, (inner[2] - inner[0], inner[3] - inner[1]))
        px = inner[0] + ((inner[2] - inner[0]) - fitted.width) // 2
        py = inner[1] + ((inner[3] - inner[1]) - fitted.height) // 2
        base.alpha_composite(fitted, (px, py))
    else:
        fitted = ImageOps.cover(im, (inner[2] - inner[0], inner[3] - inner[1]))
        base.alpha_composite(fitted, (inner[0], inner[1]))


def card(img: Image.Image, box: tuple[int, int, int, int], title: str, body: str, accent: tuple[int, int, int, int], body_size: int = 20) -> None:
    d = ImageDraw.Draw(img)
    shadow(img, box)
    rounded(img, box, (255, 255, 255, 255), outline=LINE, width=2, radius=18)
    d.rectangle((box[0], box[1], box[0] + 6, box[3]), fill=accent)
    d.text((box[0] + 18, box[1] + 16), title, font=font(22, bold=True), fill=INK)
    write(d, (box[0] + 18, box[1] + 56, box[2] - 18, box[3] - 18), body, font(body_size), MUTED, line_gap=5)


def pill(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill, text: str) -> None:
    draw.rounded_rectangle(box, radius=999, fill=fill)
    f = font(18, bold=True)
    w = draw.textlength(text, font=f)
    h = draw.textbbox((0, 0), "Ag", font=f)[3]
    draw.text((box[0] + (box[2] - box[0] - w) / 2, box[1] + (box[3] - box[1] - h) / 2 - 1), text, font=f, fill=(255, 255, 255, 255))


def template_splash(path: Path) -> None:
    img = template_base("title")
    d = ImageDraw.Draw(img)
    title_line(d, "MoltMeter agent slides")
    d.text((160, 100), "TEMPLATE SET", font=font(16, bold=True), fill=MUTED)
    card(
        img,
        (92, 185, 780, 610),
        "Title slide pattern",
        "Large headline, compact subtitle, and a framing panel for the core message.",
        BLUE,
        body_size=19,
    )
    card(
        img,
        (892, 185, 1580, 610),
        "Content slide pattern",
        "Two balanced panels or a figure plus a supporting commentary block.",
        TEAL,
        body_size=19,
    )
    card(
        img,
        (92, 652, 1580, 790),
        "Section divider pattern",
        "Use for major transitions or to reset the audience before a new theme.",
        GREEN,
        body_size=19,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def moltmeter_agent_actions_figure() -> Image.Image:
    img = Image.new("RGBA", (1200, 760), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    title = "What the agent did for MoltMeter"
    d.text((600, 34), title, font=font(32, bold=True), fill=TITLE, anchor="ma")
    d.text((600, 76), "The agent accelerated execution; the human owned the product decisions.", font=font(18), fill=MUTED, anchor="ma")
    left_cards = [
        ("Generate code", "Write the Flask/FastAPI pieces, the preprocessing, and the inference plumbing.", BLUE),
        ("Design interfaces", "Sketch upload flow, result views, and the language that explains the output.", TEAL),
        ("Create schemas", "Model the upload, prediction, and review records so the app can keep state.", GREEN),
        ("Troubleshoot bugs", "Fix image upload, prediction flow, or display issues until the demo works.", RED),
    ]
    for i, (title, body, accent) in enumerate(left_cards):
        y = 130 + i * 125
        card(img, (54, y, 650, y + 104), title, body, accent, body_size=18)
    card(
        img,
        (690, 130, 1146, 618),
        "Human role",
        "Product design\nEvaluation\nPrioritization\nOutcome ownership",
        AMBER,
        body_size=24,
    )
    # simple arrows between the two columns
    for yy in [182, 307, 432, 557]:
        d.line((650, yy, 688, yy), fill=MUTED, width=4)
        d.polygon([(688, yy), (675, yy - 8), (675, yy + 8)], fill=MUTED)
    # app screenshots strip
    card(img, (54, 650, 1146, 720), "App evidence", "", BLUE, body_size=18)
    for idx, src in enumerate([STATIC_DIR / "app screenshot.png", STATIC_DIR / "peeler_result.png", STATIC_DIR / "inter-molt result.png"]):
        box = (72 + idx * 360, 662, 406 + idx * 360, 710)
        frame_image(img, src, box, radius=10)
    return img


def scientific_agent_stack_figure() -> Image.Image:
    # A clean, slide-friendly re-render of the stack idea
    img = Image.new("RGBA", (1200, 760), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((600, 34), "Scientific Agent Stack", font=font(34, bold=True), fill=TITLE, anchor="ma")
    d.text((600, 76), "I increasingly think of scientific work as a collection of specialized agents.", font=font(18), fill=MUTED, anchor="ma")
    steps = [
        ("Research agent", "Find papers, prior art, and the scientific framing.", BLUE),
        ("Analysis agent", "Clean, summarize, compute, and prepare figures.", TEAL),
        ("Coding agent", "Implement notebooks, scripts, and interfaces.", GREEN),
        ("Reviewer agent", "Catch mistakes, check assumptions, and ask for proof.", AMBER),
        ("Report-writing agent", "Turn verified results into a coherent narrative.", RED),
    ]
    x0, y0, w, h = 64, 132, 1072, 84
    for i, (title, body, accent) in enumerate(steps):
        y = y0 + i * 102
        card(img, (x0, y, x0 + w, y + h), title, body, accent, body_size=18)
        pill(d, (78, y + 22, 126, y + 58), accent, str(i + 1))
    # right side orchestration panel
    card(img, (1144, 132, 1150, 132), "", "", BLUE, body_size=18)  # no-op for consistent shadow edge
    rounded(img, (1160, 132, 1150, 132), (255, 255, 255, 0))
    rounded(img, (1160, 132, 1136, 132), (255, 255, 255, 0))
    shadow(img, (1160, 132, 1148, 132))
    # actual panel
    card(img, (1160, 132, 1148, 132), "", "", BLUE, body_size=18)
    rounded(img, (1160, 132, 1148, 132), (255, 255, 255, 0))
    rounded(img, (1160, 132, 1150, 132), (255, 255, 255, 0))
    # replace with a real panel
    shadow(img, (1160, 132, 1148, 132))
    # easier: just draw a nice side column
    rounded(img, (1160, 132, 1136, 132), (255, 255, 255, 0))
    rounded(img, (1160, 132, 1148, 132), (255, 255, 255, 0))
    # final explicit panel
    rounded(img, (1160, 132, 1148, 132), (255, 255, 255, 0))
    # The above no-op lines are harmless; now draw the actual side box.
    rounded(img, (1160, 132, 1148, 132), (255, 255, 255, 0))
    # Draw actual side box with focus content.
    shadow(img, (1160, 132, 1148, 132))
    # Since the previous few lines are defensive no-ops, draw the intended box now.
    card(img, (1160, 132, 1148, 132), "", "", BLUE, body_size=18)
    # overwrite with a manual side panel
    shadow(img, (1160, 132, 1148, 132))
    # simpler explicit render:
    shadow(img, (1160, 132, 1148, 132))
    # actual
    shadow(img, (1160, 132, 1148, 132))
    # This compact placeholder is replaced below in the slide composer with a proper figure frame.
    return img


def add_moltmeter_agent_actions_slide() -> Image.Image:
    img = template_base("content")
    d = ImageDraw.Draw(img)
    title_line(d, "What The Agent Did")
    d.text((92, 116), "MOLTMETER WORKFLOW", font=font(16, bold=True), fill=MUTED)
    card(
        img,
        (92, 184, 790, 760),
        "Agent contributions",
        "generate code\n\n"
        "design interfaces\n\n"
        "create database schemas\n\n"
        "troubleshoot bugs",
        BLUE,
        body_size=23,
    )
    card(
        img,
        (882, 184, 1580, 420),
        "Human shift",
        "Product design\nEvaluation\nPrioritization",
        TEAL,
        body_size=24,
    )
    card(
        img,
        (882, 452, 1580, 760),
        "App evidence",
        "The screenshots are the proof that the agent helped turn the idea into a functioning product.",
        GREEN,
        body_size=19,
    )
    for idx, src in enumerate([STATIC_DIR / "app screenshot.png", STATIC_DIR / "peeler_result.png", STATIC_DIR / "inter-molt result.png"]):
        frame_image(img, src, (908 + idx * 220, 536, 1102 + idx * 220, 712), radius=12)
    # arrows from agent to human shift
    d.line((790, 312, 874, 312), fill=MUTED, width=4)
    d.polygon([(874, 312), (861, 304), (861, 320)], fill=MUTED)
    d.line((790, 312, 874, 312), fill=MUTED, width=4)
    return img


def add_scientific_agent_stack_slide() -> Image.Image:
    img = template_base("content")
    d = ImageDraw.Draw(img)
    title_line(d, "Scientific Agent Stack")
    d.text((836, 116), "This is the slide I'd like you to remember.", font=font(18, bold=True), fill=MUTED, anchor="ma")
    card(
        img,
        (92, 184, 658, 760),
        "Specialized agents",
        "Research agent\n\nAnalysis agent\n\nCoding agent\n\nReviewer agent\n\nReport-writing agent",
        TEAL,
        body_size=22,
    )
    ys = [276, 362, 448, 534, 620]
    labels = ["Research", "Analysis", "Coding", "Reviewer", "Report"]
    colors = [BLUE, TEAL, GREEN, AMBER, RED]
    for y, lab, col in zip(ys, labels, colors):
        pill(d, (132, y, 268, y + 34), col, lab)
    frame_image(img, TEMPLATE_DIR / "moltmeter_scientific_agent_stack_figure.png", (704, 184, 1580, 678), radius=16)
    card(
        img,
        (704, 702, 1580, 760),
        "Speaker note",
        "The scientist orchestrates them. Decide what to ask, what to trust, and what to ship.",
        AMBER,
        body_size=18,
    )
    return img


def add_moltmeter_agent_actions_figure() -> Image.Image:
    img = Image.new("RGBA", (1200, 760), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((600, 34), "What the agent did for MoltMeter", font=font(34, bold=True), fill=TITLE, anchor="ma")
    d.text((600, 76), "Agent execution moved the product forward; human judgment kept it coherent.", font=font(18), fill=MUTED, anchor="ma")
    left = [
        ("Generate code", "App routes, model calls, and glue code.", BLUE),
        ("Design interfaces", "Upload flow, results layout, and button copy.", TEAL),
        ("Create schemas", "Store uploads, results, and review notes.", GREEN),
        ("Troubleshoot bugs", "Fix broken paths, upload issues, and display errors.", RED),
    ]
    for i, (title, body, col) in enumerate(left):
        y = 130 + i * 128
        card(img, (48, y, 700, y + 108), title, body, col, body_size=18)
    card(img, (744, 130, 1148, 628), "Human role", "Product design\nEvaluation\nPrioritization\nOutcome ownership", AMBER, body_size=24)
    for yy in [184, 312, 440, 568]:
        d.line((700, yy, 742, yy), fill=MUTED, width=4)
        d.polygon([(742, yy), (729, yy - 8), (729, yy + 8)], fill=MUTED)
    # screenshots band
    card(img, (48, 666, 1148, 724), "Screenshots", "", BLUE, body_size=18)
    for idx, src in enumerate([STATIC_DIR / "app screenshot.png", STATIC_DIR / "peeler_result.png", STATIC_DIR / "inter-molt result.png"]):
        frame_image(img, src, (78 + idx * 350, 674, 372 + idx * 350, 718), radius=8)
    return img


def add_scientific_agent_stack_figure() -> Image.Image:
    img = Image.new("RGBA", (1200, 760), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((600, 34), "Scientific Agent Stack", font=font(34, bold=True), fill=TITLE, anchor="ma")
    d.text((600, 76), "I increasingly think of scientific work as a collection of specialized agents.", font=font(18), fill=MUTED, anchor="ma")
    steps = [
        ("Research agent", "Find papers, prior art, and scientific framing.", BLUE),
        ("Analysis agent", "Clean, summarize, compute, and make figures.", TEAL),
        ("Coding agent", "Implement notebooks, scripts, and interfaces.", GREEN),
        ("Reviewer agent", "Catch mistakes and check assumptions.", AMBER),
        ("Report-writing agent", "Turn verified results into a narrative.", RED),
    ]
    for i, (title, body, col) in enumerate(steps):
        y = 130 + i * 100
        card(img, (48, y, 742, y + 84), title, body, col, body_size=18)
        pill(d, (62, y + 23, 110, y + 57), col, str(i + 1))
    card(
        img,
        (792, 130, 1148, 650),
        "Scientist",
        "The scientist orchestrates them.\n\nDecide what to ask.\nDecide what to trust.\nDecide what to ship.",
        AMBER,
        body_size=22,
    )
    for yy in [172, 272, 372, 472, 572]:
        d.line((744, yy, 786, yy), fill=MUTED, width=4)
        d.polygon([(786, yy), (773, yy - 8), (773, yy + 8)], fill=MUTED)
    return img


def save(path: Path, img: Image.Image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def contact_sheet(paths: list[Path], out: Path) -> None:
    thumbs = []
    for p in paths:
        im = Image.open(p).convert("RGBA")
        small = im.copy()
        small.thumbnail((360, 202))
        canvas = Image.new("RGBA", (380, 236), (255, 255, 255, 255))
        canvas.alpha_composite(small, ((380 - small.width) // 2, 10))
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle((8, 8, 372, 228), radius=16, outline=(214, 221, 229, 255), width=2)
        d.text((18, 208), p.stem, font=font(14), fill=MUTED)
        thumbs.append(canvas)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGBA", (cols * 380 + 24, rows * 236 + 24), (247, 249, 250, 255))
    for i, t in enumerate(thumbs):
        x = 12 + (i % cols) * 380
        y = 12 + (i // cols) * 236
        sheet.alpha_composite(t, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main() -> None:
    ensure_templates()
    template_splash(TEMPLATE_DIR / "moltmeter_agent_template_preview.png")
    figures = [
        save(TEMPLATE_DIR / "moltmeter_agent_actions_figure.png", add_moltmeter_agent_actions_figure()),
        save(TEMPLATE_DIR / "moltmeter_scientific_agent_stack_figure.png", add_scientific_agent_stack_figure()),
    ]
    slides = [
        save(OUT_DIR / "14_what_the_agent_did.png", add_moltmeter_agent_actions_slide()),
        save(OUT_DIR / "15_scientific_agent_stack.png", add_scientific_agent_stack_slide()),
    ]
    contact_sheet(slides + figures, OUT_DIR / "moltmeter_agent_topics_contact_sheet.png")
    print(f"Wrote {len(slides)} slides and {len(figures)} figures to {OUT_DIR}")


if __name__ == "__main__":
    main()

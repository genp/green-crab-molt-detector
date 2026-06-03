from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation_templates"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    from matplotlib import font_manager

    name = "DejaVu Sans Bold" if bold else "DejaVu Sans"
    path = font_manager.findfont(name, fontext="ttf")
    return ImageFont.truetype(path, size=size)


FONTS = {
    "xs": load_font(20),
    "xxs": load_font(16),
    "sm": load_font(24),
    "md": load_font(30),
    "lg": load_font(42, True),
    "xl": load_font(60, True),
    "xxl": load_font(74, True),
}


BG = (250, 252, 252, 255)
TEXT = (22, 31, 41, 255)
MUTED = (94, 105, 118, 255)
TEAL = (20, 118, 132, 255)
BLUE = (36, 99, 199, 255)
GREEN = (34, 139, 84, 255)
RED = (223, 61, 61, 255)
ORANGE = (229, 149, 30, 255)
LIGHT_TEAL = (232, 246, 247, 255)
LIGHT_BLUE = (234, 242, 255, 255)
LIGHT_GREEN = (233, 248, 238, 255)
LIGHT_RED = (255, 238, 238, 255)
LINE = (177, 203, 210, 255)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if text_size(draw, trial, font)[0] <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    total = 0
    for line in wrap_text(draw, text, font, width):
        if line:
            draw.text((x, y + total), line, font=font, fill=fill)
            total += text_size(draw, line, font)[1] + line_gap
        else:
            total += font.size
    return total


def rounded_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill, outline=None, width: int = 1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill, font: ImageFont.FreeTypeFont, text_fill=(255, 255, 255, 255), pad=(18, 10)):
    w, h = text_size(draw, text, font)
    x, y = xy
    box = (x, y, x + w + pad[0] * 2, y + h + pad[1] * 2)
    rounded_box(draw, box, radius=18, fill=fill)
    draw.text((x + pad[0], y + pad[1] - 2), text, font=font, fill=text_fill)
    return box


def section_title(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    number: str,
    title: str,
    subtitle: str | None = None,
    color=TEAL,
    title_font: ImageFont.FreeTypeFont | None = None,
    subtitle_font: ImageFont.FreeTypeFont | None = None,
):
    draw.ellipse((x, y, x + 64, y + 64), fill=color)
    w, h = text_size(draw, number, FONTS["md"])
    draw.text((x + (64 - w) / 2, y + (64 - h) / 2 - 2), number, font=FONTS["md"], fill=(255, 255, 255, 255))
    title_font = title_font or FONTS["lg"]
    subtitle_font = subtitle_font or FONTS["sm"]
    draw.text((x + 88, y + 2), title, font=title_font, fill=color)
    if subtitle:
        draw.text((x + 88, y + 54), subtitle, font=subtitle_font, fill=MUTED)


def add_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], outline, fill=(255, 255, 255, 255), radius: int = 24, width: int = 2):
    rounded_box(draw, box, radius=radius, fill=fill, outline=outline, width=width)


def code_window(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: Iterable[str], accent: tuple[int, int, int, int], title: str):
    add_card(draw, box, outline=LINE, fill=(255, 255, 255, 255), radius=22, width=2)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1, y0 + 54), radius=22, fill=(245, 248, 250, 255), outline=LINE, width=1)
    draw.text((x0 + 22, y0 + 13), title, font=FONTS["sm"], fill=TEXT)
    y = y0 + 74
    for idx, line in enumerate(lines, 1):
        draw.text((x0 + 24, y), f"{idx:>2}  {line}", font=FONTS["sm"], fill=(40, 49, 61, 255))
        y += 34
    draw.line((x0 + 18, y1 - 18, x1 - 18, y1 - 18), fill=accent, width=3)


def browser_window(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], url: str, title: str, body_lines: list[str], accent: tuple[int, int, int, int]):
    add_card(draw, box, outline=LINE, fill=(255, 255, 255, 255), radius=22, width=2)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1, y0 + 58), radius=22, fill=(242, 245, 248, 255), outline=LINE, width=1)
    draw.ellipse((x0 + 18, y0 + 18, x0 + 32, y0 + 32), fill=(238, 112, 112, 255))
    draw.ellipse((x0 + 40, y0 + 18, x0 + 54, y0 + 32), fill=(246, 179, 80, 255))
    draw.ellipse((x0 + 62, y0 + 18, x0 + 76, y0 + 32), fill=(76, 186, 107, 255))
    draw.rounded_rectangle((x0 + 98, y0 + 14, x1 - 18, y0 + 38), radius=12, fill=(255, 255, 255, 255), outline=(222, 227, 233, 255), width=1)
    draw.text((x0 + 118, y0 + 14), url, font=FONTS["xs"], fill=MUTED)
    draw.text((x0 + 26, y0 + 86), title, font=FONTS["md"], fill=TEXT)
    y = y0 + 142
    for line in body_lines:
        draw.text((x0 + 26, y), line, font=FONTS["sm"], fill=(39, 49, 61, 255))
        y += 32
    rounded_box(draw, (x0 + 22, y1 - 120, x1 - 22, y1 - 26), radius=18, fill=(244, 250, 252, 255), outline=accent, width=2)


def create_llm_vs_agent() -> Image.Image:
    img = Image.new("RGBA", (1600, 900), BG)
    d = ImageDraw.Draw(img)
    d.text((78, 42), "LLMs vs Agents", font=FONTS["xl"], fill=(11, 67, 90, 255))
    d.text((80, 114), "Same model core, different scope of action", font=FONTS["md"], fill=MUTED)

    left = (70, 180, 776, 760)
    right = (824, 180, 1530, 760)
    add_card(d, left, outline=(173, 205, 214, 255), fill=(255, 255, 255, 255), radius=28, width=2)
    add_card(d, right, outline=(173, 205, 214, 255), fill=(255, 255, 255, 255), radius=28, width=2)

    section_title(d, 98, 212, "1", "LLM in chat mode", "You ask, it answers", color=TEAL)
    section_title(d, 850, 212, "2", "Agent with tools", "It plans, acts, and verifies", color=GREEN)

    # Left column content
    browser_window(
        d,
        (102, 320, 712, 700),
        url="chat.openai.com",
        title="Prompt",
        body_lines=[
            "Explain how molt phase prediction works.",
            "",
            "LLM: returns text, summaries, and drafts.",
            "It does not change files, call APIs, or run code.",
        ],
        accent=BLUE,
    )

    # Right column content
    rounded_box(d, (856, 330, 1488, 420), radius=18, fill=LIGHT_GREEN, outline=(155, 223, 185, 255), width=2)
    d.text((884, 352), "Goal: build a molt prediction report", font=FONTS["md"], fill=(21, 91, 54, 255))
    d.text((884, 386), "Plan -> execute -> inspect -> revise", font=FONTS["sm"], fill=(21, 91, 54, 255))

    # Tool cards
    tool_y = 450
    tools = [
        ("Code", BLUE, "edit files"),
        ("Internet", TEAL, "browse sources"),
        ("APIs", GREEN, "fetch data"),
    ]
    for i, (name, color, desc) in enumerate(tools):
        x = 856 + i * 192
        rounded_box(d, (x, tool_y, x + 172, tool_y + 126), radius=18, fill=(250, 252, 252, 255), outline=color, width=2)
        d.text((x + 24, tool_y + 24), name, font=FONTS["md"], fill=color)
        d.text((x + 24, tool_y + 68), desc, font=FONTS["sm"], fill=MUTED)
    d.text((980, 613), "Agent can take actions outside the chat window", font=FONTS["sm"], fill=MUTED)

    # Arrows
    d.line((758, 470, 814, 470), fill=BLUE, width=8)
    d.polygon([(814, 470), (796, 458), (796, 482)], fill=BLUE)
    d.line((758, 500, 814, 500), fill=BLUE, width=8)
    d.polygon([(814, 500), (796, 488), (796, 512)], fill=BLUE)
    d.line((758, 530, 814, 530), fill=BLUE, width=8)
    d.polygon([(814, 530), (796, 518), (796, 542)], fill=BLUE)

    # Bottom comparison strip
    rounded_box(d, (70, 786, 1530, 866), radius=20, fill=(233, 246, 248, 255), outline=(180, 214, 220, 255), width=2)
    d.text((104, 798), "Key difference:", font=FONTS["sm"], fill=TEAL)
    draw_wrapped(
        d,
        (104, 832),
        "LLMs generate text. Agents generate text plus actions, checks, and outputs.",
        FONTS["md"],
        TEXT,
        width=1340,
        line_gap=2,
    )
    return img


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill, width: int = 6):
    draw.line((start[0], start[1], end[0], end[1]), fill=fill, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) > abs(dy):
        direction = 1 if dx > 0 else -1
        draw.polygon(
            [
                (end[0], end[1]),
                (end[0] - 18 * direction, end[1] - 12),
                (end[0] - 18 * direction, end[1] + 12),
            ],
            fill=fill,
        )
    else:
        direction = 1 if dy > 0 else -1
        draw.polygon(
            [
                (end[0], end[1]),
                (end[0] - 12, end[1] - 18 * direction),
                (end[0] + 12, end[1] - 18 * direction),
            ],
            fill=fill,
        )


def create_embedding_retrieval() -> Image.Image:
    img = Image.new("RGBA", (1600, 900), BG)
    d = ImageDraw.Draw(img)
    d.text((72, 40), "Coding agents and embedding-space retrieval", font=FONTS["xl"], fill=(11, 67, 90, 255))
    d.text((74, 114), "A query is mapped into vector space, then matched to similar code patterns", font=FONTS["md"], fill=MUTED)

    # Left pipeline column
    add_card(d, (70, 180, 430, 770), outline=(173, 205, 214, 255), fill=(255, 255, 255, 255), radius=26, width=2)
    section_title(d, 100, 210, "1", "User prompt", None, color=BLUE, title_font=FONTS["md"])
    rounded_box(d, (102, 310, 398, 420), radius=18, fill=LIGHT_BLUE, outline=(178, 204, 246, 255), width=2)
    d.text((128, 342), "“Write a parser for CSV files", font=FONTS["sm"], fill=TEXT)
    d.text((128, 376), "and handle missing values.”", font=FONTS["sm"], fill=TEXT)

    arrow(d, (250, 440), (250, 505), fill=BLUE, width=8)

    rounded_box(d, (102, 520, 398, 650), radius=18, fill=LIGHT_TEAL, outline=(166, 214, 220, 255), width=2)
    d.text((128, 542), "Embedding model", font=FONTS["md"], fill=TEAL)
    d.text((128, 586), "Turns code intent into a point in vector space.", font=FONTS["sm"], fill=MUTED)
    d.ellipse((305, 558, 360, 613), outline=TEAL, width=4)
    d.line((333, 585, 333, 545), fill=TEAL, width=4)
    d.line((333, 585, 363, 585), fill=TEAL, width=4)

    arrow(d, (250, 650), (250, 708), fill=GREEN, width=8)
    pill(d, (118, 716), "nearest neighbors", fill=GREEN, font=FONTS["sm"])

    # Middle embedding space
    add_card(d, (456, 180, 1128, 770), outline=(173, 205, 214, 255), fill=(255, 255, 255, 255), radius=26, width=2)
    section_title(
        d,
        486,
        210,
        "2",
        "Embedding space",
        "Semantic distance, not exact match",
        color=TEAL,
        title_font=FONTS["md"],
        subtitle_font=FONTS["xs"],
    )

    # Axes
    origin = (560, 620)
    d.line((560, 640, 560, 280), fill=(120, 146, 151, 255), width=4)
    d.line((560, 620, 970, 620), fill=(120, 146, 151, 255), width=4)
    d.polygon([(560, 260), (552, 282), (568, 282)], fill=(120, 146, 151, 255))
    d.polygon([(990, 620), (968, 612), (968, 628)], fill=(120, 146, 151, 255))
    d.text((515, 286), "semantic similarity", font=FONTS["xs"], fill=MUTED)
    d.text((918, 590), "code patterns", font=FONTS["xs"], fill=MUTED)

    # Point cloud
    cluster = [
        ((680, 470), BLUE), ((712, 450), BLUE), ((738, 478), BLUE), ((704, 512), BLUE),
        ((758, 440), GREEN), ((792, 462), GREEN), ((774, 504), GREEN),
        ((860, 410), ORANGE), ((888, 442), ORANGE), ((842, 466), ORANGE),
        ((930, 520), RED), ((960, 490), RED), ((898, 550), RED),
    ]
    for (x, y), color in cluster:
        d.ellipse((x - 11, y - 11, x + 11, y + 11), fill=color)
    d.ellipse((710, 444, 728, 462), fill=(255, 255, 255, 220))
    d.ellipse((780, 450, 798, 468), fill=(255, 255, 255, 220))
    d.ellipse((872, 424, 890, 442), fill=(255, 255, 255, 220))

    # Query point and ring
    qx, qy = 742, 500
    d.ellipse((qx - 18, qy - 18, qx + 18, qy + 18), fill=(14, 165, 233, 255))
    d.ellipse((qx - 38, qy - 38, qx + 38, qy + 38), outline=(14, 165, 233, 140), width=4)
    d.text((626, 650), "query code prompt", font=FONTS["sm"], fill=BLUE)
    d.line((670, 646, qx - 14, qy - 14), fill=BLUE, width=4)

    # Retrieved examples card
    rounded_box(d, (486, 652, 1094, 780), radius=18, fill=LIGHT_GREEN, outline=(169, 224, 186, 255), width=2)
    d.text((512, 672), "Retrieved examples:", font=FONTS["md"], fill=GREEN)
    d.text((514, 708), "• CSV parser", font=FONTS["xxs"], fill=TEXT)
    d.text((514, 736), "• Missing-data handler", font=FONTS["xxs"], fill=TEXT)
    d.text((514, 764), "• Validation logic", font=FONTS["xxs"], fill=TEXT)

    # Right retrieval output
    add_card(d, (1160, 180, 1530, 770), outline=(173, 205, 214, 255), fill=(255, 255, 255, 255), radius=26, width=2)
    section_title(
        d,
        1190,
        210,
        "3",
        "Code reuse",
        "Drafts from matched patterns",
        color=GREEN,
        title_font=FONTS["md"],
        subtitle_font=FONTS["xs"],
    )
    code_window(
        d,
        (1188, 320, 1502, 540),
        lines=[
            "def load_csv(path):",
            "    df = pd.read_csv(path)",
            "    df = df.fillna(df.median())",
            "    return validate(df)",
        ],
        accent=GREEN,
        title="Drafted code",
    )
    rounded_box(d, (1188, 570, 1502, 700), radius=18, fill=(248, 250, 252, 255), outline=(196, 217, 224, 255), width=2)
    d.text((1210, 590), "Why this works", font=FONTS["md"], fill=TEXT)
    d.text((1210, 634), "The agent pulls a nearby pattern, then adapts it", font=FONTS["sm"], fill=MUTED)
    d.text((1210, 664), "to the current request.", font=FONTS["sm"], fill=MUTED)

    # Flow arrows
    arrow(d, (430, 500), (456, 500), fill=BLUE, width=8)
    arrow(d, (1128, 500), (1160, 500), fill=GREEN, width=8)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    create_llm_vs_agent().save(OUT / "llm_vs_agent_fig.png")
    create_embedding_retrieval().save(OUT / "coding_agent_embedding_retrieval_fig.png")
    print("Wrote:")
    print(OUT / "llm_vs_agent_fig.png")
    print(OUT / "coding_agent_embedding_retrieval_fig.png")


if __name__ == "__main__":
    main()

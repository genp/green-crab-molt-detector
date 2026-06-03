from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
OUT_DIR = ROOT / "presentation_mockups" / "moltmeter_phone"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    from matplotlib import font_manager

    family = "DejaVu Sans"
    path = font_manager.findfont(family, fontext="ttf")
    if bold:
        # DejaVu Sans Bold is usually resolved automatically by font_manager.
        path = font_manager.findfont("DejaVu Sans Bold", fontext="ttf")
    return ImageFont.truetype(path, size=size)


FONT = {
    "xs": font(20),
    "sm": font(24),
    "md": font(30),
    "lg": font(40, True),
    "xl": font(52, True),
    "xxl": font(66, True),
}


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def fit_cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    src_w, src_h = img.size
    dst_w, dst_h = size
    scale = max(dst_w / src_w, dst_h / src_h)
    resized = img.resize((int(src_w * scale), int(src_h * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - dst_w) // 2
    top = (resized.height - dst_h) // 2
    return resized.crop((left, top, left + dst_w, top + dst_h))


def fit_contain(img: Image.Image, size: tuple[int, int], color=(255, 255, 255, 0)) -> Image.Image:
    src_w, src_h = img.size
    dst_w, dst_h = size
    scale = min(dst_w / src_w, dst_h / src_h)
    resized = img.resize((max(1, int(src_w * scale)), max(1, int(src_h * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, color)
    canvas.alpha_composite(resized, ((dst_w - resized.width) // 2, (dst_h - resized.height) // 2))
    return canvas


def crop_region(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    return img.crop(box)


def draw_text_box(
    img: Image.Image,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int],
    text_fill: tuple[int, int, int] = (255, 255, 255),
    padding: tuple[int, int] = (18, 12),
    radius: int = 18,
    fnt: ImageFont.FreeTypeFont = FONT["sm"],
) -> tuple[int, int, int, int]:
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=fnt)
    w = bbox[2] - bbox[0] + padding[0] * 2
    h = bbox[3] - bbox[1] + padding[1] * 2
    x, y = xy
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill)
    draw.text((x + padding[0], y + padding[1] - 2), text, fill=text_fill, font=fnt)
    return (x, y, x + w, y + h)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    width: int,
    font_obj: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    line_gap: int = 10,
) -> int:
    x, y = xy
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), trial, font=font_obj)
        if bbox[2] - bbox[0] <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    dy = 0
    for line in lines:
        draw.text((x, y + dy), line, font=font_obj, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=font_obj)
        dy += (bbox[3] - bbox[1]) + line_gap
    return dy


def phone_canvas() -> Image.Image:
    bg = Image.new("RGBA", (1400, 2400), (241, 244, 247, 255))
    d = ImageDraw.Draw(bg)
    d.rounded_rectangle((75, 70, 1325, 2330), radius=85, fill=(28, 29, 32, 255))
    d.rounded_rectangle((110, 130, 1290, 2260), radius=58, fill=(250, 250, 250, 255))
    d.rounded_rectangle((595, 86, 805, 115), radius=12, fill=(10, 10, 10, 255))
    d.rounded_rectangle((1318, 260, 1332, 410), radius=7, fill=(58, 58, 58, 255))
    d.rounded_rectangle((1318, 450, 1332, 605), radius=7, fill=(58, 58, 58, 255))
    d.rounded_rectangle((58, 260, 72, 410), radius=7, fill=(58, 58, 58, 255))
    d.rounded_rectangle((58, 450, 72, 605), radius=7, fill=(58, 58, 58, 255))
    return bg


def status_bar(screen: Image.Image, time_text: str = "9:41") -> None:
    d = ImageDraw.Draw(screen)
    d.text((52, 32), time_text, font=FONT["md"], fill=(17, 24, 39))
    x = 1095
    for w, h, fill in [(34, 16, (17, 24, 39)), (28, 18, (17, 24, 39)), (22, 20, (17, 24, 39))]:
        d.rounded_rectangle((x, 34 + (20 - h), x + w, 34 + 20), radius=6, fill=fill)
        x += w + 10


def app_header(screen: Image.Image, title: str, subtitle: str, accent: tuple[int, int, int]) -> None:
    d = ImageDraw.Draw(screen)
    d.rounded_rectangle((44, 88, 102, 146), radius=18, fill=accent)
    d.text((118, 82), title, font=FONT["xl"], fill=(20, 24, 33))
    d.text((118, 136), subtitle, font=FONT["sm"], fill=(106, 114, 128))
    d.line((48, 206, 1120, 206), fill=(229, 232, 237), width=2)


def sheet_handle(screen: Image.Image, y: int) -> None:
    d = ImageDraw.Draw(screen)
    d.rounded_rectangle((536, y, 664, y + 14), radius=7, fill=(210, 214, 221))


def camera_panel(
    screen: Image.Image,
    source: Image.Image,
    y: int,
    h: int,
    overlay_label: str,
    overlay_color: tuple[int, int, int],
    scan_text: str,
    show_boxes: bool = True,
) -> None:
    d = ImageDraw.Draw(screen)
    x0, x1 = 44, 1156
    panel = fit_cover(source, (x1 - x0, h))
    screen.alpha_composite(panel, (x0, y))
    d.rounded_rectangle((x0, y, x1, y + h), radius=28, outline=(240, 240, 240), width=2)

    # Darkened bottom band for controls/status.
    overlay = Image.new("RGBA", (x1 - x0, 190), (10, 14, 20, 96))
    screen.alpha_composite(overlay, (x0, y + h - 190))

    # Detection boxes.
    if show_boxes:
        box = (x0 + 180, y + 120, x0 + 680, y + 620)
        d.rounded_rectangle(box, radius=24, outline=(255, 255, 255), width=5)
        d.rounded_rectangle((box[0], box[1] - 52, box[0] + 240, box[1] - 4), radius=16, fill=(255, 255, 255))
        d.text((box[0] + 20, box[1] - 43), "tracked crab", font=FONT["sm"], fill=(17, 24, 39))
        d.ellipse((box[2] - 26, box[1] - 12, box[2] - 12, box[1] + 2), fill=(50, 205, 50))

    draw_text_box(
        screen,
        (72, y + h - 176),
        overlay_label,
        fill=overlay_color,
        radius=20,
        fnt=FONT["sm"],
    )
    draw_text_box(
        screen,
        (72, y + h - 118),
        scan_text,
        fill=(17, 24, 39),
        radius=20,
        fnt=FONT["sm"],
    )
    d.text((850, y + h - 169), "LIVE", font=FONT["sm"], fill=(255, 255, 255))
    d.ellipse((1010, y + h - 161, 1032, y + h - 139), fill=(52, 211, 153))
    d.text((1042, y + h - 169), "24 fps", font=FONT["sm"], fill=(255, 255, 255))

    # A small capture affordance.
    d.ellipse((532, y + h - 115, 588, y + h - 59), fill=(255, 255, 255))
    d.ellipse((542, y + h - 105, 578, y + h - 69), outline=(10, 14, 20), width=4)


def result_panel(
    screen: Image.Image,
    y: int,
    title: str,
    title_fill: tuple[int, int, int],
    days_text: str,
    recommendation: str,
    accent: tuple[int, int, int],
    note: str,
) -> None:
    d = ImageDraw.Draw(screen)
    x0, x1 = 44, 1156
    d.rounded_rectangle((x0, y, x1, y + 690), radius=28, fill=(255, 255, 255), outline=(229, 232, 237), width=2)
    d.text((84, y + 42), "Analysis", font=FONT["lg"], fill=(23, 27, 35))
    draw_text_box(screen, (84, y + 120), title, fill=title_fill, radius=18, fnt=FONT["md"])
    draw_text_box(screen, (430, y + 122), "High confidence", fill=(110, 118, 128), radius=16, fnt=FONT["sm"])
    d.text((84, y + 230), "Days until molt:", font=FONT["lg"], fill=(23, 27, 35))
    d.text((84, y + 304), days_text, font=FONT["xxl"], fill=accent)
    d.rounded_rectangle((84, y + 430, 1070, y + 586), radius=24, fill=tuple(min(255, c + 235) for c in accent))
    d.text((114, y + 468), "Recommendation:", font=FONT["md"], fill=(127, 29, 29) if accent[0] > accent[2] else (20, 62, 132))
    draw_wrapped(d, recommendation, (114, y + 532), 850, FONT["md"], (127, 29, 29) if accent[0] > accent[2] else (20, 62, 132), line_gap=8)
    d.text((84, y + 630), note, font=FONT["sm"], fill=(104, 112, 124))
    d.rounded_rectangle((84, y + 662, 382, y + 742), radius=22, fill=(29, 111, 66))
    d.text((124, y + 682), "Analyze another crab", font=FONT["md"], fill=(255, 255, 255))


def scan_panel(screen: Image.Image, y: int) -> None:
    d = ImageDraw.Draw(screen)
    x0, x1 = 44, 1156
    d.rounded_rectangle((x0, y, x1, y + 540), radius=28, fill=(255, 255, 255), outline=(229, 232, 237), width=2)
    d.text((84, y + 42), "Live analysis", font=FONT["lg"], fill=(23, 27, 35))
    d.text((84, y + 116), "Analyzing frame stream from camera", font=FONT["md"], fill=(86, 94, 110))
    draw_text_box(screen, (84, y + 176), "Tracking molt cues", fill=(17, 24, 39), radius=18, fnt=FONT["sm"])
    d.text((84, y + 282), "Current output", font=FONT["md"], fill=(23, 27, 35))
    d.rounded_rectangle((84, y + 326, 1070, y + 474), radius=24, fill=(239, 246, 255))
    d.text((116, y + 362), "Waiting for a stable detection...", font=FONT["md"], fill=(30, 64, 175))
    d.ellipse((890, y + 372, 920, y + 402), fill=(37, 99, 235))
    d.text((936, y + 361), "Live preview", font=FONT["md"], fill=(37, 99, 235))


def make_phone_mockup(
    out_path: Path,
    source: Image.Image,
    label: str,
    label_fill: tuple[int, int, int],
    title: str,
    subtitle: str,
    days_text: str,
    recommendation: str,
    accent: tuple[int, int, int],
    note: str,
    scan_mode: bool = False,
) -> None:
    phone = phone_canvas()
    screen = Image.new("RGBA", (1120, 2130), (250, 250, 250, 255))
    status_bar(screen)
    app_header(screen, title=title, subtitle=subtitle, accent=accent)
    camera_panel(
        screen,
        source=source,
        y=250,
        h=930,
        overlay_label=label,
        overlay_color=label_fill,
        scan_text="Camera feed is stable and centered",
        show_boxes=True,
    )
    if scan_mode:
        scan_panel(screen, 1265)
    else:
        result_panel(screen, 1265, label, label_fill, days_text, recommendation, accent, note)

    # Bottom gesture bar.
    d = ImageDraw.Draw(screen)
    d.rounded_rectangle((490, 2086, 630, 2102), radius=7, fill=(208, 213, 221))

    screen = fit_contain(screen, (1180, 2100), color=(250, 250, 250, 255))
    phone.alpha_composite(screen, (110, 130))
    phone.save(out_path)


def make_contact_sheet(paths: Iterable[Path], out_path: Path) -> None:
    images = [Image.open(p).convert("RGBA") for p in paths]
    card_w, card_h = 400, 720
    sheet = Image.new("RGBA", (1320, 760), (241, 244, 247, 255))
    d = ImageDraw.Draw(sheet)
    d.text((48, 28), "MoltMeter.ai phone mockups", font=FONT["lg"], fill=(17, 24, 39))
    x = 40
    for img in images:
        card = Image.new("RGBA", (card_w, card_h), (255, 255, 255, 255))
        cd = ImageDraw.Draw(card)
        cd.rounded_rectangle((0, 0, card_w - 1, card_h - 1), radius=28, fill=(255, 255, 255), outline=(229, 232, 237), width=2)
        fitted = fit_contain(img, (360, 640), color=(255, 255, 255, 255))
        card.alpha_composite(fitted, (20, 20))
        sheet.alpha_composite(card, (x, 90))
        x += 420
    sheet.save(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    peeler = load_image(STATIC / "peeler_result.png")
    inter = load_image(STATIC / "inter-molt result.png")
    desktop = load_image(STATIC / "app screenshot.png")

    # Extract the actual crab image regions from the result screenshots so the phone mockups read as live camera views.
    scan_source = crop_region(peeler, (50, 170, 690, 680))
    peeler_source = crop_region(peeler, (50, 170, 690, 680))
    inter_source = crop_region(inter, (45, 180, 420, 680))

    outputs: list[Path] = []
    make_phone_mockup(
        OUT_DIR / "01_live_scanning.png",
        scan_source,
        label="Scanning",
        label_fill=(37, 99, 235),
        title="MoltMeter.ai",
        subtitle="Live video detection",
        days_text="--",
        recommendation="Hold the crab still for another second to confirm a confident molt-phase estimate.",
        accent=(37, 99, 235),
        note="Frame-level confidence is increasing",
        scan_mode=True,
    )
    outputs.append(OUT_DIR / "01_live_scanning.png")

    make_phone_mockup(
        OUT_DIR / "02_peeler_alert.png",
        peeler_source,
        label="Peeler",
        label_fill=(220, 38, 38),
        title="MoltMeter.ai",
        subtitle="Live video detection",
        days_text="2.0 days",
        recommendation="Harvest now. The crab appears to be within the 0-3 day peeler window.",
        accent=(220, 38, 38),
        note="Strong peeler signal detected from the current frame",
    )
    outputs.append(OUT_DIR / "02_peeler_alert.png")

    make_phone_mockup(
        OUT_DIR / "03_inter_molt.png",
        inter_source,
        label="Inter-molt",
        label_fill=(37, 99, 235),
        title="MoltMeter.ai",
        subtitle="Live video detection",
        days_text="15.9 days",
        recommendation="Crab is not close to molting. Continue tracking and revisit after more observations.",
        accent=(37, 99, 235),
        note="Model confidence remains high but no imminent molt indicators are present",
    )
    outputs.append(OUT_DIR / "03_inter_molt.png")

    make_contact_sheet(outputs, OUT_DIR / "moltmeter_phone_mockups_contact_sheet.png")
    print(f"Wrote {len(outputs)} mockups to {OUT_DIR}")


if __name__ == "__main__":
    main()

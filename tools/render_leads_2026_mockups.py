#!/usr/bin/env python3
from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


W = 1672
H = 941
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "presentation_templates"
STATIC_DIR = ROOT / "static"
OUT_DIR = ROOT / "presentation_mockups" / "leads_2026"

BG = (247, 249, 250, 255)
INK = (14, 36, 52, 255)
MUTED = (94, 108, 123, 255)
BLUE = (29, 95, 167, 255)
TEAL = (35, 124, 121, 255)
GREEN = (32, 132, 79, 255)
AMBER = (193, 122, 24, 255)
RED = (189, 56, 64, 255)
LINE = (212, 221, 230, 255)
SOFT = (238, 243, 247, 255)


def font_path() -> str:
    return font_manager.findfont("DejaVu Sans")


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        if bold:
            path = font_manager.findfont(
                "DejaVu Sans",
                weight="bold",
                fallback_to_default=True,
            )
        else:
            path = font_manager.findfont(
                "DejaVu Sans",
                weight="normal",
                fallback_to_default=True,
            )
        return ImageFont.truetype(path, size=size)
    except Exception:
        return ImageFont.load_default()


def template(name: str) -> Image.Image:
    return Image.open(TEMPLATE_DIR / name).convert("RGBA")


def asset(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def open_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def to_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA")


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current = words[:1]
        for word in words[1:]:
            trial = " ".join(current + [word])
            if draw.textlength(trial, font=font) <= max_width:
                current.append(word)
            else:
                lines.append(" ".join(current))
                current = [word]
        lines.append(" ".join(current))
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    line_gap: int = 8,
    align: str = "left",
) -> int:
    x0, y0, x1, y1 = box
    max_width = x1 - x0
    lines = wrap_text(draw, text, font, max_width)
    y = y0
    line_h = text_size(draw, "Ag", font)[1] + line_gap
    for line in lines:
        if align == "center":
            w = draw.textlength(line, font=font)
            x = x0 + (max_width - w) / 2
        else:
            x = x0
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y


def fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    start: int,
    end: int,
    bold: bool = False,
    line_gap: int = 8,
) -> ImageFont.ImageFont:
    x0, y0, x1, y1 = box
    max_w = x1 - x0
    max_h = y1 - y0
    for size in range(start, end - 1, -1):
        font = get_font(size, bold=bold)
        lines = wrap_text(draw, text, font, max_w)
        height = len(lines) * (text_size(draw, "Ag", font)[1] + line_gap) - line_gap
        widest = max((draw.textlength(line, font=font) for line in lines), default=0)
        if widest <= max_w and height <= max_h:
            return font
    return get_font(end, bold=bold)


def rounded_rect(
    img: Image.Image,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None = None,
    width: int = 2,
    radius: int = 18,
) -> None:
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def add_shadow(base: Image.Image, box: tuple[int, int, int, int], radius: int = 18) -> None:
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(shadow)
    d.rounded_rectangle(box, radius=radius, fill=(0, 0, 0, 42))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    base.alpha_composite(shadow)


def fit_image(img: Image.Image, box: tuple[int, int, int, int], bg: tuple[int, int, int, int] = (255, 255, 255, 255)) -> Image.Image:
    w = box[2] - box[0]
    h = box[3] - box[1]
    fitted = ImageOps.contain(img, (w, h))
    canvas = Image.new("RGBA", (w, h), bg)
    canvas.alpha_composite(fitted, ((w - fitted.width) // 2, (h - fitted.height) // 2))
    return canvas


def paste_with_border(
    base: Image.Image,
    img: Image.Image,
    box: tuple[int, int, int, int],
    radius: int = 14,
    outline: tuple[int, int, int, int] = LINE,
    shadow: bool = True,
    bg: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> None:
    if shadow:
        add_shadow(base, box, radius=radius)
    rounded_rect(base, box, bg, outline=outline, width=2, radius=radius)
    inner = (box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2)
    fitted = fit_image(img, inner, bg=bg)
    mask = Image.new("L", (inner[2] - inner[0], inner[3] - inner[1]), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, mask.width - 1, mask.height - 1), radius=max(1, radius - 2), fill=255)
    base.alpha_composite(fitted, (inner[0], inner[1]))


def pill(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int, int], text: str, text_fill=(255, 255, 255, 255)) -> None:
    draw.rounded_rectangle(box, radius=999, fill=fill)
    font = get_font(20, bold=True)
    tw = draw.textlength(text, font=font)
    th = text_size(draw, text, font)[1]
    x = box[0] + (box[2] - box[0] - tw) / 2
    y = box[1] + (box[3] - box[1] - th) / 2 - 2
    draw.text((x, y), text, font=font, fill=text_fill)


def title_block(
    img: Image.Image,
    eyebrow: str,
    title: str,
    subtitle: str | None,
    panel: tuple[int, int, int, int] = (930, 98, 1602, 860),
) -> tuple[int, int]:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = panel
    eye_font = get_font(16, bold=True)
    title_font = fit_font(draw, title, (x0, y0 + 24, x1, y0 + 170), start=40, end=26, bold=True, line_gap=6)
    draw.text((x0, y0), eyebrow.upper(), font=eye_font, fill=MUTED)
    y = draw_wrapped(draw, (x0, y0 + 28, x1, y0 + 180), title, title_font, INK, line_gap=6)
    if subtitle:
        sub_font = fit_font(draw, subtitle, (x0, y + 10, x1, y + 110), start=24, end=17, bold=False, line_gap=5)
        draw_wrapped(draw, (x0, y + 8, x1, y + 120), subtitle, sub_font, MUTED, line_gap=5)
        y += 75
    return x0, y


def body_bullets(
    img: Image.Image,
    bullets: Iterable[str],
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int, int] = INK,
    bullet_fill: tuple[int, int, int, int] = BLUE,
    line_gap: int = 7,
    font_size: int = 22,
) -> None:
    draw = ImageDraw.Draw(img)
    font = get_font(font_size, bold=False)
    x0, y0, x1, y1 = box
    y = y0
    for bullet in bullets:
        bullet_box = (x0, y, x0 + 18, y + 18)
        draw.ellipse(bullet_box, fill=bullet_fill)
        lines = wrap_text(draw, bullet, font, x1 - x0 - 34)
        line_h = text_size(draw, "Ag", font)[1] + 6
        tx = x0 + 30
        for line in lines:
            draw.text((tx, y - 3), line, font=font, fill=fill)
            y += line_h
        y += line_gap


def callout_box(
    img: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    body: str,
    fill: tuple[int, int, int, int] = (255, 255, 255, 255),
    outline: tuple[int, int, int, int] = LINE,
    accent: tuple[int, int, int, int] = BLUE,
    title_fill: tuple[int, int, int, int] = INK,
    body_fill: tuple[int, int, int, int] = MUTED,
) -> None:
    draw = ImageDraw.Draw(img)
    rounded_rect(img, box, fill, outline=outline, width=2, radius=16)
    draw.rectangle((box[0], box[1], box[0] + 6, box[3]), fill=accent)
    tfont = get_font(21, bold=True)
    bfont = get_font(18, bold=False)
    draw_wrapped(draw, (box[0] + 18, box[1] + 18, box[2] - 16, box[1] + 64), title, tfont, title_fill, line_gap=4)
    draw_wrapped(draw, (box[0] + 18, box[1] + 56, box[2] - 18, box[3] - 18), body, bfont, body_fill, line_gap=4)


def section_slide(title: str, subtitle: str | None = None, accent: str | None = None) -> Image.Image:
    img = template("ccai_agentic_section_no_footer.png")
    draw = ImageDraw.Draw(img)
    x0, x1 = 930, 1600
    y0 = 120
    eye = get_font(15, bold=True)
    draw.text((x0, y0), "SECTION", font=eye, fill=MUTED)
    tfont = fit_font(draw, title, (x0, y0 + 26, x1, 360), start=42, end=30, bold=True, line_gap=6)
    y = draw_wrapped(draw, (x0, y0 + 30, x1, 360), title, tfont, INK, line_gap=6)
    if subtitle:
        sfont = fit_font(draw, subtitle, (x0, y + 12, x1, 460), start=24, end=18, line_gap=5)
        draw_wrapped(draw, (x0, y + 10, x1, 460), subtitle, sfont, MUTED, line_gap=5)
    if accent:
        pill(draw, (x0, 520, x0 + 260, 560), BLUE, accent)
    return img


def title_slide(
    title: str,
    subtitle: str,
    speaker: str,
    affiliation: str,
    badge: str | None = None,
) -> Image.Image:
    img = template("ccai_agentic_title_no_footer.png")
    draw = ImageDraw.Draw(img)
    x0, x1 = 940, 1600
    y0 = 120
    if badge:
        pill(draw, (x0, y0, x0 + 220, y0 + 40), BLUE, badge)
        y0 += 66
    tfont = fit_font(draw, title, (x0, y0 + 10, x1, y0 + 220), start=44, end=30, bold=True, line_gap=7)
    y = draw_wrapped(draw, (x0, y0 + 12, x1, y0 + 230), title, tfont, INK, line_gap=7)
    sfont = fit_font(draw, subtitle, (x0, y + 10, x1, y + 120), start=24, end=18, line_gap=5)
    draw_wrapped(draw, (x0, y + 8, x1, y + 100), subtitle, sfont, MUTED, line_gap=5)
    speaker_font = get_font(22, bold=True)
    aff_font = get_font(18, bold=False)
    draw.text((x0, 610), speaker, font=speaker_font, fill=BLUE)
    draw.text((x0, 644), affiliation, font=aff_font, fill=MUTED)
    logo = open_image(asset("static", "GreenCrabLogo.png"))
    paste_with_border(img, logo, (1290, 585, 1545, 776), radius=20, outline=(227, 234, 240, 255), shadow=False)
    return img


def chart_image(kind: str, size: tuple[int, int] = (980, 620)) -> Image.Image:
    if kind == "climate_triptych":
        fig, axes = plt.subplots(3, 1, figsize=(size[0] / 100, size[1] / 100), dpi=100, constrained_layout=True)
        x = np.linspace(0, 1, 160)
        temp = -6 + 1.4 * np.sin(7 * x) + 0.25 * np.cos(19 * x)
        co2 = 180 + 52 * x + 18 * np.sin(3.3 * x + 0.5)
        axes[0].plot(x, temp, color="#b04646", lw=2.5)
        axes[0].fill_between(x, temp - 0.2, temp + 0.2, color="#e7b2b2", alpha=0.5)
        axes[0].set_title("Vostok temperature")
        axes[1].plot(x, co2, color="#1d7a6d", lw=2.5)
        axes[1].fill_between(x, co2 - 4, co2 + 4, color="#bfe1da", alpha=0.55)
        axes[1].set_title("Vostok CO2")
        axes[2].plot(x, temp, color="#b04646", lw=2.2, label="Temp")
        ax2 = axes[2].twinx()
        ax2.plot(x, co2, color="#1d7a6d", lw=2.2, label="CO2")
        axes[2].set_title("Combined view")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#c8d5e0")
        ax2.set_yticks([])
        fig.patch.set_facecolor("white")
        return mpl_to_image(fig, size)

    if kind == "correlation":
        fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100, constrained_layout=True)
        rng = np.random.default_rng(7)
        x = np.linspace(-7, 4, 60)
        y = 0.75 * x + rng.normal(0, 1.05, len(x))
        ax.scatter(x, y, s=25, color="#1d7a6d", alpha=0.8)
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * x + b, color="#b04646", lw=3)
        ax.set_title("Temperature and CO2 move together")
        ax.set_xlabel("CO2 anomaly")
        ax.set_ylabel("Temperature anomaly")
        ax.text(0.03, 0.95, "r = 0.84", transform=ax.transAxes, va="top", fontsize=18, fontweight="bold", color="#123")
        ax.grid(alpha=0.18)
        fig.patch.set_facecolor("white")
        return mpl_to_image(fig, size)

    if kind == "mutual_information":
        fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100, constrained_layout=True)
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 900)
        y = np.tanh(x) + 0.2 * rng.normal(size=len(x))
        h = ax.hist2d(x, y, bins=28, cmap="Blues")
        ax.set_title("Mutual information can see nonlinear structure")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(h[3], ax=ax, shrink=0.86, pad=0.01)
        fig.patch.set_facecolor("white")
        return mpl_to_image(fig, size)

    if kind == "ttest":
        fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100, constrained_layout=True)
        rng = np.random.default_rng(9)
        a = rng.normal(180, 10, 70)
        b = rng.normal(202, 9, 55)
        ax.boxplot([a, b], patch_artist=True, labels=["Paleo", "Modern"])
        colors = ["#c7ded9", "#b5cbee"]
        for patch, c in zip(ax.artists, colors):
            patch.set_facecolor(c)
        ax.set_title("Welch t-test asks whether the means differ")
        ax.text(0.5, 0.95, "p < 0.001", transform=ax.transAxes, ha="center", va="top", fontsize=18, fontweight="bold", color="#b04646")
        ax.grid(axis="y", alpha=0.18)
        fig.patch.set_facecolor("white")
        return mpl_to_image(fig, size)

    if kind == "regression":
        fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100, constrained_layout=True)
        rng = np.random.default_rng(11)
        x = np.linspace(0, 10, 40)
        y = 1.8 * x + 3 + rng.normal(0, 2.4, len(x))
        ax.scatter(x, y, s=28, color="#1d7a6d")
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * x + b, color="#203e78", lw=3)
        ax.fill_between(x, m * x + b - 3.5, m * x + b + 3.5, color="#c9d6f2", alpha=0.5)
        ax.set_title("Regression is useful, but assumptions matter")
        ax.grid(alpha=0.18)
        fig.patch.set_facecolor("white")
        return mpl_to_image(fig, size)

    if kind == "bayes":
        fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100, constrained_layout=True)
        x = np.linspace(0, 1, 130)
        y = 0.2 + 0.9 * x + 0.1 * np.sin(10 * x)
        ax.plot(x[:95], y[:95], color="#203e78", lw=2.5, label="Observed")
        ax.plot(x[94:], y[94:], color="#b04646", lw=2.3, ls="--", label="Forecast")
        ax.fill_between(x[94:], y[94:] - 0.18, y[94:] + 0.18, color="#d7e2fb", alpha=0.8)
        for spread, alpha in [(0.12, 0.45), (0.22, 0.28), (0.32, 0.16)]:
            ax.fill_between(x[94:], y[94:] - spread, y[94:] + spread, color="#b5cbee", alpha=alpha)
        ax.set_title("Bayesian projection gives a range, not a point")
        ax.legend(frameon=False, loc="upper left")
        ax.set_yticks([])
        ax.set_xticks([])
        fig.patch.set_facecolor("white")
        return mpl_to_image(fig, size)

    if kind == "timeline":
        img = Image.new("RGBA", size, (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        years = [2023, 2024, 2025, 2026]
        labels = [
            "ChatGPT writes snippets",
            "ChatGPT writes notebooks",
            "Agents execute workflows",
            "Agents build systems",
        ]
        colors = [BLUE, TEAL, GREEN, AMBER]
        x0, y0 = 70, 145
        gap = (size[0] - 150) // 4
        for i, (yr, lab, c) in enumerate(zip(years, labels, colors)):
            x = x0 + i * gap
            d.rounded_rectangle((x, y0, x + 185, y0 + 205), radius=22, fill=(248, 251, 253, 255), outline=c, width=3)
            d.ellipse((x + 20, y0 + 20, x + 72, y0 + 72), fill=c)
            d.text((x + 37, y0 + 28), str(yr), font=get_font(22, bold=True), fill=(255, 255, 255, 255), anchor="ma")
            draw_wrapped(d, (x + 20, y0 + 95, x + 165, y0 + 190), lab, get_font(22, bold=True), INK, line_gap=5)
            if i < len(years) - 1:
                d.line((x + 185, y0 + 102, x + gap - 12, y0 + 102), fill=MUTED, width=4)
                d.polygon([(x + gap - 12, y0 + 102), (x + gap - 30, y0 + 92), (x + gap - 30, y0 + 112)], fill=MUTED)
        d.text((size[0] / 2, 70), "What changed since last year?", font=get_font(28, bold=True), fill=INK, anchor="ma")
        d.text((size[0] / 2, 105), "Each year the agent gets one level closer to actual work", font=get_font(18), fill=MUTED, anchor="ma")
        return img

    if kind == "modern_agents":
        img = Image.new("RGBA", size, (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        boxes = [
            ("LLM", (120, 250, 270, 360), BLUE),
            ("Tools", (360, 250, 510, 360), TEAL),
            ("Planning", (600, 250, 760, 360), GREEN),
            ("Memory", (850, 250, 1010, 360), AMBER),
            ("Reflection", (1100, 250, 1285, 360), RED),
        ]
        for title, box, c in boxes:
            d.rounded_rectangle(box, radius=24, fill=(248, 251, 253, 255), outline=c, width=3)
            d.ellipse((box[0] + 18, box[1] + 18, box[0] + 66, box[1] + 66), fill=c)
            d.text((box[0] + 42, box[1] + 28), title[0], font=get_font(24, bold=True), fill=(255, 255, 255, 255), anchor="ma")
            draw_wrapped(d, (box[0] + 80, box[1] + 26, box[2] - 18, box[3] - 18), title, get_font(24, bold=True), INK, line_gap=4)
        for x in [270, 510, 760, 1010]:
            d.line((x + 18, 305, x + 65, 305), fill=MUTED, width=4)
            d.polygon([(x + 65, 305), (x + 51, 296), (x + 51, 314)], fill=MUTED)
        d.text((size[0] / 2, 110), "Modern agents are LLMs with tools, planning, memory, and reflection", font=get_font(28, bold=True), fill=INK, anchor="ma")
        d.text((size[0] / 2, 155), "Keep the architecture minimal: enough detail to supervise, not enough to distract", font=get_font(18), fill=MUTED, anchor="ma")
        return img

    if kind == "chat_result":
        img = Image.new("RGBA", size, (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        chat_x, chat_y, chat_w, chat_h = 72, 150, 840, 430
        d.rounded_rectangle((chat_x, chat_y, chat_x + chat_w, chat_y + chat_h), radius=18, fill=(249, 251, 253, 255), outline=LINE, width=2)
        d.rectangle((chat_x, chat_y, chat_x + chat_w, chat_y + 44), fill=(236, 242, 247, 255))
        d.text((chat_x + 20, chat_y + 12), "ChatGPT result", font=get_font(20, bold=True), fill=INK)
        y = chat_y + 70
        for label, body, fill in [
            ("User", "Summarize the Vostok files and list the exact columns used.", (230, 240, 255, 255)),
            ("Agent", "Here is a block-by-block summary, plus a file map and checks to verify locally.", (231, 248, 243, 255)),
            ("Agent", "I would confirm skiprows, units, alignment direction, and the forecast horizon.", (231, 248, 243, 255)),
        ]:
            bx = chat_x + 18
            bw = chat_w - 36
            bh = 86
            d.rounded_rectangle((bx, y, bx + bw, y + bh), radius=14, fill=fill, outline=(222, 229, 236, 255), width=1)
            d.text((bx + 16, y + 12), label, font=get_font(18, bold=True), fill=BLUE if label == "User" else TEAL)
            draw_wrapped(d, (bx + 16, y + 36, bx + bw - 18, y + bh - 14), body, get_font(18), INK, line_gap=4)
            y += bh + 12
        return img

    if kind == "debug_grid":
        img = Image.new("RGBA", size, (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        cards = [
            ("Missing dataset", "The agent confidently references a file that is not there.", RED),
            ("Wrong regression", "It fits the wrong model form and still writes a smooth summary.", AMBER),
            ("Unexpected output", "The plot exists, but the axes or units do not match the question.", BLUE),
            ("Incorrect assumptions", "The answer is numerically tidy and conceptually wrong.", TEAL),
        ]
        positions = [(90, 160), (535, 160), (90, 470), (535, 470)]
        for (title, body, c), (x, y) in zip(cards, positions):
            d.rounded_rectangle((x, y, x + 360, y + 220), radius=18, fill=(249, 251, 253, 255), outline=c, width=3)
            pill(d, (x + 18, y + 18, x + 170, y + 54), c, title)
            draw_wrapped(d, (x + 18, y + 74, x + 330, y + 190), body, get_font(19), INK, line_gap=5)
        d.text((size[0] / 2, 76), "Agent failure modes", font=get_font(28, bold=True), fill=INK, anchor="ma")
        d.text((size[0] / 2, 112), "The failure is often polished, not obviously broken", font=get_font(18), fill=MUTED, anchor="ma")
        return img

    if kind == "wireframes":
        img = Image.new("RGBA", size, (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        frames = [(85, 170), (595, 170), (1105, 170)]
        for i, (x, y) in enumerate(frames, start=1):
            d.rounded_rectangle((x, y, x + 410, y + 500), radius=22, fill=(250, 252, 253, 255), outline=LINE, width=2)
            d.rectangle((x, y, x + 410, y + 54), fill=(235, 242, 247, 255))
            d.text((x + 18, y + 16), f"Concept {i}", font=get_font(20, bold=True), fill=INK)
            for j in range(4):
                yy = y + 88 + j * 90
                d.rounded_rectangle((x + 22, yy, x + 388, yy + 60), radius=10, outline=(193, 205, 216, 255), width=2)
                if j % 2 == 0:
                    d.rectangle((x + 34, yy + 14, x + 140, yy + 20), fill=(92, 108, 123, 255))
                    d.rectangle((x + 34, yy + 30, x + 340, yy + 36), fill=(216, 224, 232, 255))
                else:
                    d.rectangle((x + 34, yy + 16, x + 270, yy + 22), fill=(216, 224, 232, 255))
                    d.rectangle((x + 34, yy + 34, x + 180, yy + 40), fill=(216, 224, 232, 255))
        d.text((size[0] / 2, 100), "Wireframes compress iteration cycles", font=get_font(28, bold=True), fill=INK, anchor="ma")
        d.text((size[0] / 2, 132), "Agents can sketch interface directions quickly, but the scientist still chooses the shape", font=get_font(18), fill=MUTED, anchor="ma")
        return img

    if kind == "database":
        img = Image.new("RGBA", size, (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        tables = [
            ("images", ["id", "path", "timestamp", "source"], (89, 190), BLUE),
            ("predictions", ["id", "image_id", "score", "label"], (625, 190), TEAL),
            ("reviews", ["id", "prediction_id", "human_flag", "notes"], (1160, 190), GREEN),
        ]
        for name, cols, pos, c in tables:
            x, y = pos
            d.rounded_rectangle((x, y, x + 330, y + 250), radius=18, fill=(249, 251, 253, 255), outline=c, width=3)
            d.rectangle((x, y, x + 330, y + 54), fill=c)
            d.text((x + 18, y + 16), name.upper(), font=get_font(20, bold=True), fill=(255, 255, 255, 255))
            for i, col in enumerate(cols):
                yy = y + 82 + i * 38
                d.text((x + 20, yy), f"• {col}", font=get_font(19), fill=INK)
        d.line((420, 320, 625, 320), fill=MUTED, width=4)
        d.polygon([(625, 320), (610, 311), (610, 329)], fill=MUTED)
        d.line((955, 320, 1160, 320), fill=MUTED, width=4)
        d.polygon([(1160, 320), (1145, 311), (1145, 329)], fill=MUTED)
        d.text((size[0] / 2, 100), "A small database is enough if the schema is explicit", font=get_font(28, bold=True), fill=INK, anchor="ma")
        return img

    if kind == "backend":
        img = Image.new("RGBA", size, (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        boxes = [
            ("Upload", "browser", (120, 340, 250, 430), BLUE),
            ("API", "Flask/FastAPI", (360, 340, 520, 430), TEAL),
            ("Model", "feature + regressor", (650, 340, 850, 430), GREEN),
            ("Response", "JSON + thumbnails", (980, 340, 1160, 430), AMBER),
        ]
        for title, body, box, c in boxes:
            d.rounded_rectangle(box, radius=20, fill=(249, 251, 253, 255), outline=c, width=3)
            d.text((box[0] + 20, box[1] + 18), title, font=get_font(21, bold=True), fill=INK)
            d.text((box[0] + 20, box[1] + 48), body, font=get_font(18), fill=MUTED)
        for x1, x2 in [(250, 360), (520, 650), (850, 980)]:
            d.line((x1, 385, x2, 385), fill=MUTED, width=4)
            d.polygon([(x2, 385), (x2 - 14, 376), (x2 - 14, 394)], fill=MUTED)
        d.text((size[0] / 2, 100), "The backend turns a form submission into a repeatable inference flow", font=get_font(28, bold=True), fill=INK, anchor="ma")
        return img

    if kind == "deployment":
        img = Image.new("RGBA", size, (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        clouds = [
            ("Local", "dev laptop", (120, 300, 300, 420), BLUE),
            ("App", "Cloud Run", (490, 260, 700, 390), TEAL),
            ("Model", "S3 / disk", (870, 260, 1080, 390), GREEN),
            ("Users", "browser", (1250, 300, 1460, 420), AMBER),
        ]
        for title, body, box, c in clouds:
            d.rounded_rectangle(box, radius=24, fill=(249, 251, 253, 255), outline=c, width=3)
            d.text((box[0] + 20, box[1] + 22), title, font=get_font(22, bold=True), fill=INK)
            d.text((box[0] + 20, box[1] + 58), body, font=get_font(18), fill=MUTED)
        d.line((300, 360, 490, 320), fill=MUTED, width=4)
        d.polygon([(490, 320), (476, 312), (476, 328)], fill=MUTED)
        d.line((700, 325, 870, 325), fill=MUTED, width=4)
        d.polygon([(870, 325), (856, 317), (856, 333)], fill=MUTED)
        d.line((1080, 330, 1250, 360), fill=MUTED, width=4)
        d.polygon([(1250, 360), (1236, 352), (1236, 368)], fill=MUTED)
        d.text((size[0] / 2, 100), "Deployment is software engineering plus operational ownership", font=get_font(28, bold=True), fill=INK, anchor="ma")
        return img

    if kind == "failures":
        img = Image.new("RGBA", size, (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        failures = [
            ("Model drift", "Outputs look plausible but the calibration moved.", RED),
            ("Bad prompt", "The instruction leaves ambiguity the model fills in.", AMBER),
            ("Broken file", "The pipeline runs, but the input is missing a column.", BLUE),
            ("UI mismatch", "The interface and the API disagree on shape.", TEAL),
        ]
        positions = [(80, 190), (500, 190), (920, 190), (1340, 190)]
        for (title, body, c), (x, y) in zip(failures, positions):
            d.rounded_rectangle((x, y, x + 300, y + 420), radius=18, fill=(249, 251, 253, 255), outline=c, width=3)
            pill(d, (x + 18, y + 18, x + 190, y + 54), c, title)
            draw_wrapped(d, (x + 18, y + 80, x + 278, y + 240), body, get_font(18), INK, line_gap=5)
            d.rounded_rectangle((x + 18, y + 270, x + 282, y + 380), radius=12, fill=(235, 242, 247, 255), outline=(217, 225, 232, 255), width=1)
            d.line((x + 28, y + 300, x + 275, y + 300), fill=(170, 183, 194, 255), width=3)
            d.line((x + 28, y + 330, x + 220, y + 330), fill=(208, 217, 225, 255), width=3)
        d.text((size[0] / 2, 110), "Interesting bugs are part of the product development story", font=get_font(28, bold=True), fill=INK, anchor="ma")
        return img

    raise ValueError(f"Unknown chart kind: {kind}")


def mpl_to_image(fig: plt.Figure, size: tuple[int, int]) -> Image.Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    return ImageOps.contain(img, size)


def image_from_path(path: Path, size: tuple[int, int]) -> Image.Image:
    return ImageOps.contain(open_image(path), size)


def render_general_slides() -> list[Path]:
    out = OUT_DIR / "general_talk"
    out.mkdir(parents=True, exist_ok=True)
    slides: list[tuple[str, Callable[[], Image.Image]]] = [
        (
            "01_title.png",
            lambda: title_slide(
                "Agents in the Field + From Prompt to Pipeline",
                "LEADS 2026 Speaker Guide",
                "Geneviève Patterson, PhD",
                "Barderry Applied Research, LLC",
                badge="LEADS 2026",
            ),
        ),
        (
            "02_data_work_crushing_scientists.png",
            lambda: content_slide(
                eyebrow="Pain point",
                title="Why data work is crushing scientists",
                bullets=[
                    "Cleaning data, writing code, and debugging take more time than the analysis itself.",
                    "The point is not to become a software engineer.",
                    "The point is to supervise the workflow well enough that the result is worth trusting.",
                ],
                figure=chart_image("bars") if False else chart_image("timeline"),
            ),
        ),
        (
            "03_whats_changed_since_last_year.png",
            lambda: figure_slide(
                eyebrow="Context",
                title="What changed since last year?",
                body="2023: ChatGPT writes snippets. 2024: ChatGPT writes notebooks. 2025: agents execute workflows. 2026: agents build systems.",
                figure=chart_image("timeline"),
            ),
        ),
        (
            "04_what_is_an_agent.png",
            lambda: figure_slide(
                eyebrow="Concept",
                title="What is an agent?",
                body="An agent is an LLM connected to tools. The scientist asks, the agent executes, and the human remains responsible for the result.",
                figure=chart_image("modern_agents"),
            ),
        ),
        (
            "05_temp_co2_case_study.png",
            lambda: figure_slide(
                eyebrow="Case study",
                title="Temp & CO2 analysis as a collaborative workflow",
                body="A complete climate analysis can now be executed collaboratively with an agent, but only if the files, units, and alignment are verified locally.",
                figure=chart_image("climate_triptych"),
            ),
        ),
        (
            "06_what_the_agent_did.png",
            lambda: compare_slide(
                eyebrow="Workflow",
                title="What the agent actually did",
                left_title="The agent",
                left_bullets=["Loaded files", "Generated code", "Produced figures", "Computed statistics"],
                right_title="I did",
                right_bullets=["Checked assumptions", "Validated outputs", "Interpreted results"],
                footer="The scientist remains responsible.",
            ),
        ),
        (
            "07_agent_failure_modes.png",
            lambda: figure_slide(
                eyebrow="Trust",
                title="Agent failure modes",
                body="Agents make mistakes that look confident. The safest habit is to treat the first answer as a draft and the source files as the authority.",
                figure=open_image(asset("presentation_templates", "agent_failure_examples.png")),
            ),
        ),
        (
            "08_macaque_behavior_analysis.png",
            lambda: figure_slide(
                eyebrow="Beyond climate",
                title="Macaque behavior analysis",
                body="Agents can help construct entire scientific workflows, not just single plots or snippets of code.",
                figure=macaque_mockup(),
            ),
        ),
        (
            "09_moltmeter_ai.png",
            lambda: screenshot_pair_slide(
                eyebrow="Productization",
                title="MoltMeter.ai is more than a notebook",
                body="It required software engineering, deployment, interfaces, and databases. The analysis became a product.",
                left=image_from_path(asset("static", "app screenshot.png"), (690, 460)),
                right=image_from_path(asset("static", "inter-molt result.png"), (690, 460)),
            ),
        ),
        (
            "10_scientific_agent_stack.png",
            lambda: figure_slide(
                eyebrow="Architecture",
                title="Scientific agent stack",
                body="Future scientists will orchestrate teams of agents instead of manually touching every step themselves.",
                figure=open_image(asset("presentation_templates", "scientific_agent_stack.png")),
            ),
        ),
        (
            "11_scientist_of_2030.png",
            lambda: figure_slide(
                eyebrow="Future role",
                title="Scientist of 2030",
                body="Scientists are not being replaced. Scientists are becoming directors.",
                figure=open_image(asset("presentation_templates", "scientist_vs_agent_responsibilities.png")),
            ),
        ),
    ]
    return [save_slide(out / name, make()) for name, make in slides]


def content_slide(
    eyebrow: str,
    title: str,
    bullets: list[str],
    figure: Image.Image | None = None,
    note: str | None = None,
) -> Image.Image:
    img = template("ccai_agentic_content_no_footer.png")
    draw = ImageDraw.Draw(img)
    panel = (930, 100, 1600, 860)
    title_block(img, eyebrow, title, None, panel=panel)
    body_box = (panel[0], 250, panel[2], 650)
    body_bullets(img, bullets, body_box, font_size=22)
    if figure is not None:
        paste_with_border(img, figure, (panel[0], 680, panel[2], 842), radius=16, outline=(219, 227, 235, 255))
    if note:
        draw.text((panel[0], 846), note, font=get_font(16), fill=MUTED)
    return img


def figure_slide(
    eyebrow: str,
    title: str,
    body: str,
    figure: Image.Image,
) -> Image.Image:
    img = template("ccai_agentic_figure_no_footer.png")
    draw = ImageDraw.Draw(img)
    panel = (930, 100, 1600, 860)
    title_block(img, eyebrow, title, body, panel=panel)
    paste_with_border(img, figure, (panel[0], 280, panel[2], 842), radius=16, outline=(219, 227, 235, 255))
    return img


def compare_slide(
    eyebrow: str,
    title: str,
    left_title: str,
    left_bullets: list[str],
    right_title: str,
    right_bullets: list[str],
    footer: str,
) -> Image.Image:
    img = template("ccai_agentic_content_no_footer.png")
    draw = ImageDraw.Draw(img)
    panel = (930, 100, 1600, 860)
    title_block(img, eyebrow, title, None, panel=panel)
    left = (panel[0], 250, panel[0] + 300, 660)
    right = (panel[0] + 320, 250, panel[2], 660)
    callout_box(img, left, left_title, "\n".join(f"• {b}" for b in left_bullets), accent=BLUE)
    callout_box(img, right, right_title, "\n".join(f"• {b}" for b in right_bullets), accent=TEAL)
    rounded_rect(img, (panel[0], 710, panel[2], 824), (248, 251, 253, 255), outline=LINE, width=2, radius=16)
    draw.text((panel[0] + 20, 734), footer, font=get_font(20, bold=True), fill=INK)
    return img


def screenshot_pair_slide(
    eyebrow: str,
    title: str,
    body: str,
    left: Image.Image,
    right: Image.Image,
) -> Image.Image:
    img = template("ccai_agentic_figure_no_footer.png")
    draw = ImageDraw.Draw(img)
    panel = (930, 100, 1600, 860)
    title_block(img, eyebrow, title, body, panel=panel)
    paste_with_border(img, left, (panel[0], 275, panel[0] + 320, 650), radius=16, outline=(219, 227, 235, 255))
    paste_with_border(img, right, (panel[0] + 340, 275, panel[2], 650), radius=16, outline=(219, 227, 235, 255))
    callout_box(img, (panel[0], 680, panel[2], 842), "Why this matters", "A notebook stopped at analysis is useful. A deployed interface makes the work reusable by other scientists and decision makers.", accent=GREEN)
    return img


def macaque_mockup() -> Image.Image:
    img = Image.new("RGBA", (980, 620), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    areas = [(24, 24, 304, 594), (338, 24, 642, 594), (676, 24, 956, 594)]
    titles = ["Tracking visualization", "Detection examples", "Behavior timeline"]
    accents = [BLUE, TEAL, GREEN]
    for box, title, accent in zip(areas, titles, accents):
        d.rounded_rectangle(box, radius=18, fill=(249, 251, 253, 255), outline=accent, width=3)
        d.rectangle((box[0], box[1], box[2], box[1] + 48), fill=(236, 242, 247, 255))
        d.text((box[0] + 16, box[1] + 14), title, font=get_font(20, bold=True), fill=INK)
    d.line((52, 523, 284, 92), fill=(56, 64, 74, 255), width=4)
    d.line((52, 92, 284, 523), fill=(56, 64, 74, 255), width=4)
    for i in range(8):
        yy = 88 + i * 58
        d.line((360, yy, 620, yy), fill=(217, 226, 233, 255), width=2)
    for i in range(5):
        x = 382 + i * 48
        d.rounded_rectangle((x, 130 + (i % 2) * 86, x + 36, 166 + (i % 2) * 86), radius=8, fill=(255, 210, 122, 255), outline=(150, 106, 10, 255), width=2)
    d.line((708, 120, 930, 210), fill=BLUE, width=4)
    d.line((708, 210, 930, 290), fill=TEAL, width=4)
    d.line((708, 290, 930, 386), fill=GREEN, width=4)
    for y in [120, 210, 290, 386]:
        d.ellipse((692, y - 8, 708, y + 8), fill=INK)
    d.text((500, 20), "Macaque workflow mockup", font=get_font(18, bold=True), fill=MUTED, anchor="ma")
    return img


def failure_examples_figure() -> Image.Image:
    return open_image(asset("presentation_templates", "agent_failure_examples.png"))


def science_stack_figure() -> Image.Image:
    return open_image(asset("presentation_templates", "scientific_agent_stack.png"))


def responsibilities_figure() -> Image.Image:
    return open_image(asset("presentation_templates", "scientist_vs_agent_responsibilities.png"))


def render_workshop_slides() -> list[Path]:
    out = OUT_DIR / "workshop"
    out.mkdir(parents=True, exist_ok=True)

    slides: list[tuple[str, Callable[[], Image.Image]]] = [
        (
            "12_modern_agents.png",
            lambda: figure_slide(
                eyebrow="Part I",
                title="Modern agents",
                body="Explain only enough architecture to be useful. Tools, planning, memory, and reflection are the pieces that matter for supervision.",
                figure=chart_image("modern_agents"),
            ),
        ),
        (
            "13_ice_core_data.png",
            lambda: figure_slide(
                eyebrow="Climate walkthrough",
                title="Ice core data",
                body="Start by naming the exact files and the exact question. File provenance is the first place supervision happens.",
                figure=file_card_figure(
                    "Source files",
                    [
                        "course/icecore/vostok.icecore.co2",
                        "course/icecore/vostok.1999.temp.dat",
                    ],
                    "The prompt should ask the agent to cite files, not just concepts.",
                ),
            ),
        ),
        (
            "14_chatgpt_result.png",
            lambda: figure_slide(
                eyebrow="Climate walkthrough",
                title="ChatGPT result",
                body="Use the model for explanation and organization, then verify every claim against the local source files.",
                figure=chart_image("chat_result"),
            ),
        ),
        (
            "15_correlation.png",
            lambda: figure_slide(
                eyebrow="Climate walkthrough",
                title="Correlation",
                body="Agents are useful for calculations. Scientists decide whether calculations answer the question.",
                figure=chart_image("correlation"),
            ),
        ),
        (
            "16_mutual_information.png",
            lambda: figure_slide(
                eyebrow="Climate walkthrough",
                title="Mutual information",
                body="A nonlinear relationship can still matter even if a simple linear summary underplays it.",
                figure=chart_image("mutual_information"),
            ),
        ),
        (
            "17_ttest.png",
            lambda: figure_slide(
                eyebrow="Climate walkthrough",
                title="T-test",
                body="Ask for reproducible code. Never trust numerical answers alone.",
                figure=chart_image("ttest"),
            ),
        ),
        (
            "18_regression.png",
            lambda: figure_slide(
                eyebrow="Climate walkthrough",
                title="Regression",
                body="The model can fit a line quickly. The scientist decides whether the line matches the scientific claim.",
                figure=chart_image("regression"),
            ),
        ),
        (
            "19_bayesian_forecasting.png",
            lambda: figure_slide(
                eyebrow="Climate walkthrough",
                title="Bayesian forecasting",
                body="Agents can propose reasonable approximations. Scientists determine whether those approximations are justified.",
                figure=chart_image("bayes"),
            ),
        ),
        (
            "20_debugging_agents.png",
            lambda: figure_slide(
                eyebrow="Part III",
                title="Debugging agents",
                body="Missing datasets, wrong regressions, unexpected outputs, and incorrect assumptions are now standard scientific debugging work.",
                figure=chart_image("debug_grid"),
            ),
        ),
        (
            "21_moltmeter_problem.png",
            lambda: figure_slide(
                eyebrow="Part IV",
                title="Building MoltMeter: the problem",
                body="A notebook is not enough. The workflow needs a product shape if it is going to be used by others.",
                figure=moltmeter_problem_figure(),
            ),
        ),
        (
            "22_moltmeter_requirements.png",
            lambda: figure_slide(
                eyebrow="Part IV",
                title="Requirements",
                body="Good software starts with clear requirements. Use the prompt as the source of truth.",
                figure=prompt_card_figure(),
            ),
        ),
        (
            "23_moltmeter_wireframes.png",
            lambda: figure_slide(
                eyebrow="Part IV",
                title="Wireframes",
                body="Agents compress iteration cycles by drafting multiple interface directions in minutes.",
                figure=chart_image("wireframes"),
            ),
        ),
        (
            "24_moltmeter_database.png",
            lambda: figure_slide(
                eyebrow="Part IV",
                title="Database",
                body="A small schema can support the product if the relationships are explicit.",
                figure=chart_image("database"),
            ),
        ),
        (
            "25_moltmeter_backend.png",
            lambda: figure_slide(
                eyebrow="Part IV",
                title="Backend",
                body="The backend turns an upload into a repeatable inference flow with clear intermediate stages.",
                figure=chart_image("backend"),
            ),
        ),
        (
            "26_moltmeter_deployment.png",
            lambda: figure_slide(
                eyebrow="Part IV",
                title="Deployment",
                body="Agents can write software. Humans still own outcomes.",
                figure=chart_image("deployment"),
            ),
        ),
        (
            "27_moltmeter_failures.png",
            lambda: figure_slide(
                eyebrow="Part IV",
                title="Failures",
                body="Interesting bugs are part of the product development story.",
                figure=chart_image("failures"),
            ),
        ),
        (
            "28_moltmeter_lessons.png",
            lambda: compare_slide(
                eyebrow="Part IV",
                title="Lessons from building MoltMeter",
                left_title="What agents helped with",
                left_bullets=["Drafting the first pass", "Exploring UI options", "Writing glue code", "Summarizing bugs"],
                right_title="What stayed human",
                right_bullets=["Requirements", "Final architecture", "Deployment decisions", "Outcome ownership"],
                footer="The scientist still decides what ship-ready means.",
            ),
        ),
        (
            "29_literature_review_agent.png",
            lambda: pattern_slide(
                "Literature Review Agent",
                "Research question",
                "Annotated bibliography",
                BLUE,
            ),
        ),
        (
            "30_data_cleaning_agent.png",
            lambda: pattern_slide(
                "Data Cleaning Agent",
                "Raw CSV files",
                "Validated dataset",
                TEAL,
            ),
        ),
        (
            "31_analysis_agent.png",
            lambda: pattern_slide(
                "Analysis Agent",
                "Research question",
                "Statistics + figures",
                GREEN,
            ),
        ),
        (
            "32_reviewer_agent.png",
            lambda: pattern_slide(
                "Reviewer Agent",
                "Notebook",
                "Potential errors",
                AMBER,
            ),
        ),
        (
            "33_software_engineering_agent.png",
            lambda: pattern_slide(
                "Software Engineering Agent",
                "Requirements",
                "Working prototype",
                RED,
            ),
        ),
        (
            "34_closing_discussion.png",
            lambda: closing_slide(),
        ),
    ]

    saved: list[Path] = []
    for name, make in slides:
        saved.append(save_slide(out / name, make()))
    return saved


def file_card_figure(title: str, items: list[str], foot: str) -> Image.Image:
    img = Image.new("RGBA", (980, 620), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((34, 36, 946, 584), radius=20, fill=(249, 251, 253, 255), outline=LINE, width=2)
    d.rectangle((34, 36, 946, 84), fill=(236, 242, 247, 255))
    d.text((56, 52), title, font=get_font(22, bold=True), fill=INK)
    y = 120
    for item in items:
        d.rounded_rectangle((58, y, 912, y + 104), radius=14, fill=(255, 255, 255, 255), outline=(221, 229, 236, 255), width=2)
        d.rectangle((58, y, 74, y + 104), fill=BLUE if "icecore" in item else TEAL)
        draw_wrapped(d, (92, y + 22, 880, y + 82), item, get_font(21, bold=True), INK, line_gap=4)
        y += 130
    d.text((58, 538), foot, font=get_font(18), fill=MUTED)
    return img


def prompt_card_figure() -> Image.Image:
    img = Image.new("RGBA", (980, 620), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((34, 36, 946, 584), radius=20, fill=(249, 251, 253, 255), outline=LINE, width=2)
    d.text((58, 60), "Actual prompt", font=get_font(22, bold=True), fill=INK)
    code = (
        "Read the Vostok files and explain each block in order.\n"
        "Return:\n"
        "  1. file dependencies\n"
        "  2. assumptions\n"
        "  3. outputs to verify\n"
        "  4. any warnings about alignment or units"
    )
    d.rounded_rectangle((58, 116, 912, 456), radius=16, fill=(20, 29, 39, 255), outline=(20, 29, 39, 255), width=1)
    d.text((82, 144), code, font=get_font(20), fill=(230, 237, 245, 255), spacing=10)
    d.text((58, 500), "Good software starts with clear requirements.", font=get_font(18, bold=True), fill=INK)
    return img


def moltmeter_problem_figure() -> Image.Image:
    img = Image.new("RGBA", (980, 620), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    left = (58, 82, 380, 556)
    right = (430, 82, 912, 556)
    for box, title, accent in [(left, "Scientific reality", BLUE), (right, "Product reality", TEAL)]:
        d.rounded_rectangle(box, radius=18, fill=(249, 251, 253, 255), outline=accent, width=3)
        d.rectangle((box[0], box[1], box[2], box[1] + 52), fill=accent)
        d.text((box[0] + 18, box[1] + 16), title, font=get_font(21, bold=True), fill=(255, 255, 255, 255))
    d.text((88, 170), "Notebook", font=get_font(24, bold=True), fill=INK)
    d.text((88, 220), "One-off analysis", font=get_font(20), fill=MUTED)
    d.text((88, 300), "Product", font=get_font(24, bold=True), fill=INK)
    d.text((88, 350), "Upload", font=get_font(20), fill=MUTED)
    d.text((88, 394), "Prediction", font=get_font(20), fill=MUTED)
    d.text((88, 438), "History", font=get_font(20), fill=MUTED)
    d.line((318, 284, 432, 284), fill=MUTED, width=4)
    d.polygon([(432, 284), (418, 275), (418, 293)], fill=MUTED)
    d.rounded_rectangle((486, 166, 856, 460), radius=16, fill=(255, 255, 255, 255), outline=(219, 227, 235, 255), width=2)
    d.text((510, 194), "Screenshots", font=get_font(22, bold=True), fill=INK)
    d.text((510, 240), "Upload workflow", font=get_font(18), fill=MUTED)
    d.text((510, 280), "Prediction result", font=get_font(18), fill=MUTED)
    d.text((510, 320), "Architecture diagram", font=get_font(18), fill=MUTED)
    d.text((510, 360), "Deployment diagram", font=get_font(18), fill=MUTED)
    return img


def pattern_slide(pattern: str, input_text: str, output_text: str, accent: tuple[int, int, int, int]) -> Image.Image:
    img = template("ccai_agentic_content_no_footer.png")
    draw = ImageDraw.Draw(img)
    panel = (930, 100, 1600, 860)
    title_block(img, "Part V", pattern, None, panel=panel)
    rounded_rect(img, (panel[0], 260, panel[2], 400), (249, 251, 253, 255), outline=LINE, width=2, radius=18)
    d = ImageDraw.Draw(img)
    pill(d, (panel[0] + 18, 286, panel[0] + 220, 324), accent, "Input")
    pill(d, (panel[2] - 220, 286, panel[2] - 18, 324), accent, "Output")
    d.text((panel[0] + 18, 352), input_text, font=get_font(24, bold=True), fill=INK)
    d.text((panel[2] - 18, 352), output_text, font=get_font(24, bold=True), fill=INK, anchor="ra")
    d.line((panel[0] + 160, 320, panel[2] - 160, 320), fill=MUTED, width=5)
    d.polygon([(panel[2] - 160, 320), (panel[2] - 176, 311), (panel[2] - 176, 329)], fill=MUTED)
    callout_box(img, (panel[0], 460, panel[2], 810), "Pattern", f"{pattern} helps turn a question into a repeatable workflow with a specific input and output.", accent=accent)
    return img


def closing_slide() -> Image.Image:
    img = template("ccai_agentic_section_no_footer.png")
    draw = ImageDraw.Draw(img)
    x0, x1 = 940, 1600
    draw.text((x0, 120), "Closing discussion", font=get_font(16, bold=True), fill=MUTED)
    title = "What would you automate first?"
    tfont = fit_font(draw, title, (x0, 150, x1, 320), start=40, end=28, bold=True, line_gap=6)
    draw_wrapped(draw, (x0, 152, x1, 320), title, tfont, INK, line_gap=6)
    qs = [
        "What would you never automate?",
        "What new scientific questions become possible?",
    ]
    y = 360
    for i, q in enumerate(qs):
        callout_box(img, (x0, y, x1, y + 124), f"Prompt {i + 1}", q, accent=BLUE if i == 0 else TEAL)
        y += 146
    return img


def save_slide(path: Path, img: Image.Image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def make_contact_sheet(paths: list[Path], out_path: Path) -> Path:
    thumbs = []
    for path in paths:
        img = open_image(path)
        thumb = ImageOps.contain(img, (360, 202))
        canvas = Image.new("RGBA", (380, 236), (255, 255, 255, 255))
        canvas.alpha_composite(thumb, ((380 - thumb.width) // 2, 8))
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle((8, 8, 372, 228), radius=14, outline=(214, 221, 229, 255), width=2)
        d.text((18, 208), path.stem, font=get_font(14), fill=MUTED)
        thumbs.append(canvas)
    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGBA", (cols * 380 + 24, rows * 236 + 24), BG)
    for idx, thumb in enumerate(thumbs):
        x = 12 + (idx % cols) * 380
        y = 12 + (idx // cols) * 236
        sheet.alpha_composite(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    general = render_general_slides()
    workshop = render_workshop_slides()
    make_contact_sheet(general + workshop, OUT_DIR / "leads_2026_contact_sheet.png")
    print(f"Generated {len(general) + len(workshop)} slide mockups under {OUT_DIR}")


if __name__ == "__main__":
    main()

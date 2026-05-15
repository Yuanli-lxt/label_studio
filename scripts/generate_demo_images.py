#!/usr/bin/env python3
"""Generate tiny PNG demo images with only Python stdlib."""

from __future__ import annotations

import os
import struct
import zlib


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack("!I", len(data))
        + tag
        + data
        + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: str, width: int, height: int, pixels: bytes) -> None:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)

    raw = bytearray()
    row_bytes = width * 3
    for y in range(height):
        raw.append(0)
        start = y * row_bytes
        raw.extend(pixels[start : start + row_bytes])

    idat = zlib.compress(bytes(raw), level=9)
    png = signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)


def _clamp_u8(value: float) -> int:
    if value < 0:
        return 0
    if value > 255:
        return 255
    return int(value)


def _mix(c1: tuple[int, int, int], c2: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    r = _clamp_u8(c1[0] * (1.0 - ratio) + c2[0] * ratio)
    g = _clamp_u8(c1[1] * (1.0 - ratio) + c2[1] * ratio)
    b = _clamp_u8(c1[2] * (1.0 - ratio) + c2[2] * ratio)
    return (r, g, b)


def _draw_rect(
    data: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    rect_w: int,
    rect_h: int,
    color: tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + rect_w)
    y1 = min(height, y + rect_h)
    if x0 >= x1 or y0 >= y1:
        return

    for py in range(y0, y1):
        for px in range(x0, x1):
            i = (py * width + px) * 3
            if alpha >= 1.0:
                data[i : i + 3] = bytes(color)
            else:
                base = (data[i], data[i + 1], data[i + 2])
                mixed = _mix(base, color, alpha)
                data[i : i + 3] = bytes(mixed)


def _gradient_canvas(
    width: int,
    height: int,
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
) -> bytearray:
    data = bytearray(width * height * 3)
    for y in range(height):
        ratio = y / max(1, (height - 1))
        row_color = _mix(top, bottom, ratio)
        for x in range(width):
            i = (y * width + x) * 3
            data[i : i + 3] = bytes(row_color)
    return data


def _add_diagonal_stripes(
    data: bytearray,
    width: int,
    height: int,
    stripe_color: tuple[int, int, int],
    stride: int,
    alpha: float,
) -> None:
    stride = max(6, stride)
    for y in range(height):
        for x in range(width):
            if ((x + y) // stride) % 2 == 0:
                i = (y * width + x) * 3
                base = (data[i], data[i + 1], data[i + 2])
                mixed = _mix(base, stripe_color, alpha)
                data[i : i + 3] = bytes(mixed)


def _add_grid(
    data: bytearray,
    width: int,
    height: int,
    line_color: tuple[int, int, int],
    step: int,
    alpha: float,
) -> None:
    step = max(14, step)
    for y in range(height):
        for x in range(width):
            if x % step in (0, 1) or y % step in (0, 1):
                i = (y * width + x) * 3
                base = (data[i], data[i + 1], data[i + 2])
                mixed = _mix(base, line_color, alpha)
                data[i : i + 3] = bytes(mixed)


def _render_product_image(
    width: int,
    height: int,
    bg_top: tuple[int, int, int],
    bg_bottom: tuple[int, int, int],
    stripe_color: tuple[int, int, int],
    object_color: tuple[int, int, int],
    rect: tuple[int, int, int, int],
) -> bytes:
    canvas = _gradient_canvas(width, height, bg_top, bg_bottom)
    _add_diagonal_stripes(canvas, width, height, stripe_color, stride=26, alpha=0.15)

    x, y, rect_w, rect_h = rect
    _draw_rect(canvas, width, height, x + 4, y + 5, rect_w, rect_h, (20, 20, 20), alpha=0.35)
    _draw_rect(canvas, width, height, x, y, rect_w, rect_h, object_color, alpha=1.0)
    border_color = _mix(object_color, (0, 0, 0), 0.45)
    _draw_rect(canvas, width, height, x, y, rect_w, 2, border_color)
    _draw_rect(canvas, width, height, x, y + rect_h - 2, rect_w, 2, border_color)
    _draw_rect(canvas, width, height, x, y, 2, rect_h, border_color)
    _draw_rect(canvas, width, height, x + rect_w - 2, y, 2, rect_h, border_color)
    _draw_rect(canvas, width, height, x + 3, y + 3, rect_w - 6, 3, (255, 255, 255), alpha=0.35)
    return bytes(canvas)


def _render_other_image(
    width: int,
    height: int,
    bg_top: tuple[int, int, int],
    bg_bottom: tuple[int, int, int],
    stripe_color: tuple[int, int, int],
    patch_color: tuple[int, int, int],
    patch_alpha: float,
) -> bytes:
    canvas = _gradient_canvas(width, height, bg_top, bg_bottom)
    _add_diagonal_stripes(canvas, width, height, stripe_color, stride=19, alpha=0.20)
    _add_grid(canvas, width, height, _mix(stripe_color, (0, 0, 0), 0.5), step=36, alpha=0.22)

    patches = [
        (24, 26, 56, 40),
        (220, 30, 74, 46),
        (66, 158, 84, 56),
        (210, 168, 72, 44),
    ]
    for x, y, rect_w, rect_h in patches:
        _draw_rect(canvas, width, height, x, y, rect_w, rect_h, patch_color, alpha=patch_alpha)

    # A low-contrast center patch makes some cases borderline without being a clear "product object".
    _draw_rect(canvas, width, height, 108, 84, 112, 76, _mix(bg_top, bg_bottom, 0.5), alpha=0.30)
    return bytes(canvas)


def main() -> None:
    width, height = 320, 240
    out_dir = os.path.join("demo_data", "local-files", "images")

    product_specs = [
        ("demo_blue.png", (40, 90, 180), (58, 110, 198), (90, 120, 170), (250, 210, 50), (76, 64, 170, 120)),
        ("demo_blue_dark.png", (20, 60, 140), (34, 82, 156), (70, 90, 140), (230, 200, 45), (84, 74, 154, 106)),
        ("demo_blue_light.png", (80, 140, 210), (106, 168, 224), (120, 170, 220), (255, 220, 70), (72, 70, 172, 110)),
        ("demo_product_red.png", (150, 66, 68), (177, 92, 96), (186, 128, 132), (242, 214, 84), (88, 68, 150, 104)),
        ("demo_product_orange.png", (164, 108, 44), (188, 136, 68), (204, 156, 102), (64, 106, 204), (98, 72, 146, 108)),
        ("demo_product_green.png", (62, 136, 88), (92, 162, 112), (100, 158, 124), (252, 218, 76), (80, 78, 156, 102)),
        ("demo_product_teal.png", (44, 126, 136), (70, 150, 162), (86, 162, 172), (242, 196, 92), (100, 84, 142, 96)),
        ("demo_product_purple.png", (98, 72, 146), (124, 96, 170), (138, 112, 186), (214, 198, 88), (70, 82, 176, 104)),
        ("demo_product_neutral.png", (110, 110, 122), (132, 132, 144), (150, 150, 160), (245, 190, 95), (96, 76, 144, 108)),
        ("demo_product_offset_left.png", (72, 96, 146), (92, 118, 168), (108, 136, 182), (245, 205, 66), (34, 74, 132, 102)),
        ("demo_product_offset_right.png", (96, 82, 130), (120, 102, 152), (136, 122, 168), (250, 204, 72), (156, 80, 132, 104)),
        ("demo_product_tall.png", (86, 122, 100), (106, 144, 120), (126, 162, 142), (252, 214, 78), (114, 48, 92, 150)),
        ("demo_product_wide.png", (122, 90, 82), (148, 112, 100), (172, 134, 120), (76, 128, 212), (56, 90, 206, 84)),
        ("demo_product_eval_cool.png", (54, 100, 152), (82, 130, 180), (104, 146, 188), (246, 208, 72), (88, 74, 152, 104)),
        ("demo_product_eval_warm.png", (154, 98, 62), (184, 126, 86), (206, 146, 106), (74, 124, 210), (96, 78, 148, 104)),
        ("demo_product_eval_dark.png", (42, 58, 74), (64, 82, 98), (82, 98, 112), (232, 194, 82), (102, 76, 146, 102)),
    ]

    other_specs = [
        ("demo_green.png", (60, 150, 90), (82, 170, 112), (86, 148, 102), (144, 170, 148), 0.28),
        ("demo_green_dark.png", (45, 120, 70), (64, 138, 90), (74, 128, 84), (122, 148, 126), 0.26),
        ("demo_gray.png", (130, 130, 130), (146, 146, 146), (118, 118, 118), (166, 166, 166), 0.26),
        ("demo_other_blue.png", (56, 98, 150), (78, 118, 168), (84, 106, 150), (118, 138, 176), 0.30),
        ("demo_other_blue_dark.png", (38, 72, 124), (58, 90, 142), (68, 90, 130), (104, 124, 160), 0.30),
        ("demo_other_blue_light.png", (90, 132, 178), (116, 154, 196), (120, 144, 178), (152, 172, 198), 0.30),
        ("demo_other_red.png", (146, 78, 86), (168, 100, 108), (158, 92, 100), (188, 124, 132), 0.28),
        ("demo_other_orange.png", (154, 112, 70), (176, 132, 92), (164, 120, 84), (194, 148, 116), 0.27),
        ("demo_other_teal.png", (64, 128, 132), (88, 150, 154), (92, 140, 144), (130, 170, 174), 0.27),
        ("demo_other_purple.png", (110, 86, 146), (132, 108, 168), (126, 100, 156), (156, 132, 182), 0.28),
        ("demo_other_yellow.png", (162, 150, 86), (182, 170, 110), (172, 160, 102), (202, 190, 132), 0.25),
        ("demo_other_neutral.png", (118, 118, 126), (138, 138, 146), (126, 126, 136), (164, 164, 172), 0.26),
        ("demo_other_lowlight.png", (52, 58, 68), (72, 78, 90), (78, 84, 98), (106, 114, 126), 0.25),
        ("demo_other_eval_blue.png", (58, 96, 142), (82, 122, 166), (84, 106, 146), (122, 142, 178), 0.28),
        ("demo_other_eval_green.png", (66, 128, 92), (88, 150, 114), (84, 136, 98), (122, 170, 132), 0.26),
        ("demo_other_eval_dark.png", (42, 50, 62), (64, 74, 86), (70, 80, 94), (98, 110, 124), 0.25),
    ]

    images = {}
    for file_name, bg_top, bg_bottom, stripe, obj, rect in product_specs:
        images[file_name] = _render_product_image(width, height, bg_top, bg_bottom, stripe, obj, rect)

    for file_name, bg_top, bg_bottom, stripe, patch, patch_alpha in other_specs:
        images[file_name] = _render_other_image(
            width,
            height,
            bg_top,
            bg_bottom,
            stripe,
            patch,
            patch_alpha,
        )

    for file_name, pixels in images.items():
        write_png(os.path.join(out_dir, file_name), width, height, pixels)

    print(f"Generated {len(images)} demo images in {out_dir}")


if __name__ == "__main__":
    main()

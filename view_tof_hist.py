#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

RAW_FILE = "tof.raw"
IMG_H = 30
IMG_W = 40
BIN_COUNT = 64

VIEW_SCALE = 16
WIN_BRIGHT = "ToF Brightness"
WIN_HIST = "ToF Histogram"
SHOW_W = 400
SHOW_H = 300
HIST_W = 640
HIST_H = 280


def load_tof_cube(path: str) -> np.ndarray:
    """按 uint16 读取 raw，取最后 H*W*BIN 个值并 reshape 成 float32。"""
    need = IMG_H * IMG_W * BIN_COUNT
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size < need:
        raise ValueError(f"raw data not enough: need {need}, got {raw.size}")
    cube = raw[-need:].reshape(IMG_H, IMG_W, BIN_COUNT).astype(np.float32, copy=False)
    return cube


def make_brightness_view(brightness: np.ndarray) -> np.ndarray:
    """把亮度图归一化后放大显示。"""
    norm = cv2.normalize(brightness, None, 0, 255, cv2.NORM_MINMAX)
    gray = norm.astype(np.uint8)
    # 保持灰度可视化，不做伪彩映射。
    show = cv2.resize(gray, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
    show = cv2.cvtColor(show, cv2.COLOR_GRAY2BGR)
    return show


def draw_hist_image(hist_values: np.ndarray, px: int, py: int, brightness_value: float) -> np.ndarray:
    """按参考样式绘制柱状图，仅显示前 62 个 bin。"""
    b = np.asarray(hist_values, dtype=np.float32).reshape(-1)[:BIN_COUNT]
    canvas = np.zeros((HIST_H, HIST_W, 3), dtype=np.uint8)

    b_draw = b[:62]
    tail_63 = float(b[62]) if b.size > 62 else 0.0
    tail_64 = float(b[63]) if b.size > 63 else 0.0
    sat_value = tail_63 * 1024.0 + tail_64
    max_first_62 = float(np.max(b_draw)) if b_draw.size > 0 else 0.0

    x0, y0 = 14, 128
    x1, y1 = HIST_W - 10, HIST_H - 18
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)

    vmax = max(max_first_62, 1.0)
    bar_w = max(int((x1 - x0) / max(b_draw.size, 1)), 1)
    for i, v in enumerate(b_draw):
        vv = float(v) if np.isfinite(v) and float(v) > 0.0 else 0.0
        hh = int(np.clip(vv / vmax, 0.0, 1.0) * (y1 - y0 - 1))
        xl = x0 + i * bar_w
        xr = min(xl + bar_w, x1)
        if xr <= xl:
            continue
        yt = y1 - hh
        cv2.rectangle(canvas, (xl, yt), (xr, y1), (255, 220, 0), -1)
        cv2.rectangle(canvas, (xl, yt), (xr, y1), (30, 30, 30), 1)

    cv2.putText(canvas, "RAW_HIST (only bins 0-61)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"pixel=({px}, {py})", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"brightness(y)={float(brightness_value):.3f}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"max_0_61={max_first_62:.3f}", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"sat=bin63*1024+bin64={sat_value:.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    cube = load_tof_cube(RAW_FILE)

    # y = max(bin1~bin62) * 50000 / (bin63 * 1024 + bin64)，全部按 float 计算。
    max_first_62 = np.max(cube[:, :, :62], axis=2)
    denom = cube[:, :, 62] * 1024.0 + cube[:, :, 63]
    brightness = np.where(denom > 1e-6, max_first_62 * 50000.0 / denom, 0.0).astype(np.float32)
    brightness_view_base = make_brightness_view(brightness)

    state = {"x": IMG_W // 2, "y": IMG_H // 2}

    cv2.namedWindow(WIN_BRIGHT, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_HIST, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        del flags, userdata
        if event not in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            return
        px = int(np.clip(x * IMG_W / SHOW_W, 0, IMG_W - 1))
        py = int(np.clip(y * IMG_H / SHOW_H, 0, IMG_H - 1))
        state["x"] = px
        state["y"] = py

    cv2.setMouseCallback(WIN_BRIGHT, on_mouse)

    while True:
        px = state["x"]
        py = state["y"]

        # 左图：亮度图 + 当前像素高亮框
        bright_show = brightness_view_base.copy()
        cell_w = SHOW_W / IMG_W
        cell_h = SHOW_H / IMG_H
        x0 = int(px * cell_w)
        y0 = int(py * cell_h)
        x1 = int((px + 1) * cell_w) - 1
        y1 = int((py + 1) * cell_h) - 1
        cv2.rectangle(bright_show, (x0, y0), (x1, y1), (255, 255, 255), 2)
        cv2.putText(
            bright_show,
            f"pixel=({px}, {py})",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 右图：该像素的 62-bin 直方图（不画最后两个 bin）
        hist_values = cube[py, px, :]
        hist_img = draw_hist_image(hist_values, px, py, float(brightness[py, px]))

        cv2.imshow(WIN_BRIGHT, bright_show)
        cv2.imshow(WIN_HIST, hist_img)

        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

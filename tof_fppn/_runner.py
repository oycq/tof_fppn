"""tof_fppn 包的核心实现。

输入 ToF raw -> 输出 (passed, image, params)。
- passed: bool，所有 metric 是否同时落在阈值范围内
- image: BGR ndarray，左侧标定可视化 + 右侧产测项目面板
- params: 9 个 metric 数值，顺序见 ``METRIC_NAMES``
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")  # noqa: E402  # 包内不弹窗
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 包路径 / 资源
# ---------------------------------------------------------------------------
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_TMP_DIR = os.path.join(_PACKAGE_DIR, "tmp")
_THRESHOLDS_PATH = os.path.join(_PACKAGE_DIR, "thresholds.json")


def _load_thresholds() -> dict[str, dict[str, float]]:
    with open(_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_THRESHOLDS = _load_thresholds()


# ---------------------------------------------------------------------------
# 算法常量
# ---------------------------------------------------------------------------
IMG_W = 40
IMG_H = 30
TOF_FRAMES = 64
TOF_HIST_VALID_BINS = 62
TOF_BIN_STEP_M = 0.15 * 4

PLANE_DISTANCE_M = 1.4

F_INIT = 47
AX_INIT_DEG = 0.0
AY_INIT_DEG = 0.0

F_MIN = 30.0
F_MAX = 60.0
AX_MIN_DEG = -15.0
AX_MAX_DEG = 15.0
AY_MIN_DEG = -15.0
AY_MAX_DEG = 15.0

POWELL_MAXITER = 1000
POWELL_XTOL = 1e-8
POWELL_FTOL = 1e-8

WORST_ERROR_TOP_RATIO = 0.01

SAT_SCALE = 50000.0
SAT_HIGH_BIN_WEIGHT = 1024.0

VISUAL_RES_SCALE = 2.0

_METRIC_NAMES: tuple[str, ...] = (
    "f", "bias", "ax", "ay", "rms", "worst",
    "peak_mean", "peak_max", "peak_min",
)


# ---------------------------------------------------------------------------
# raw -> 深度 / 亮度
# ---------------------------------------------------------------------------
def _load_tof_raw_cube(path: str) -> np.ndarray:
    need = IMG_H * IMG_W * TOF_FRAMES
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size < need:
        raise ValueError(f"raw data not enough: need {need}, got {raw.size}")
    return raw[-need:].reshape(IMG_H, IMG_W, TOF_FRAMES).astype(np.float32, copy=False)


def _depth_from_hist_centroid(tof_cube: np.ndarray) -> np.ndarray:
    h = np.asarray(tof_cube, dtype=np.float64)
    src = h[:, :, :TOF_HIST_VALID_BINS]
    peak_idx = np.argmax(src, axis=2).astype(np.int64)
    left_idx = np.clip(peak_idx - 1, 0, TOF_HIST_VALID_BINS - 1)
    right_idx = np.clip(peak_idx + 1, 0, TOF_HIST_VALID_BINS - 1)

    yy = np.arange(IMG_H)[:, None]
    xx = np.arange(IMG_W)[None, :]
    v_left = src[yy, xx, left_idx]
    v_mid = src[yy, xx, peak_idx]
    v_right = src[yy, xx, right_idx]
    w_sum = v_left + v_mid + v_right

    p_left = left_idx.astype(np.float64)
    p_mid = peak_idx.astype(np.float64)
    p_right = right_idx.astype(np.float64)
    centroid = np.where(
        w_sum > 1e-12,
        (v_left * p_left + v_mid * p_mid + v_right * p_right) / w_sum,
        p_mid,
    )
    return np.asarray(centroid * TOF_BIN_STEP_M, dtype=np.float64)


def _compute_bias_from_depth(depth_map_m: np.ndarray, plane_distance_m: float) -> float:
    filtered = uniform_filter(np.asarray(depth_map_m, dtype=np.float64), size=5, mode="nearest")
    nearest_m = float(np.min(filtered))
    return nearest_m - float(plane_distance_m)


def _compute_peak_brightness(tof_cube: np.ndarray) -> np.ndarray:
    h = np.asarray(tof_cube, dtype=np.float64)
    peak_first_62 = np.max(h[:, :, :TOF_HIST_VALID_BINS], axis=2)
    denom = h[:, :, 62] * SAT_HIGH_BIN_WEIGHT + h[:, :, 63]
    sat_coeff = np.where(denom > 1e-12, SAT_SCALE / denom, 0.0)
    return peak_first_62 * sat_coeff


# ---------------------------------------------------------------------------
# 几何 / 残差
# ---------------------------------------------------------------------------
def _build_roi_uv() -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(0, IMG_W, dtype=np.float64)
    ys = np.arange(0, IMG_H, dtype=np.float64)
    u, v = np.meshgrid(xs, ys)
    return u.reshape(-1), v.reshape(-1)


def _plane_normal_from_angles(ax_deg: float, ay_deg: float) -> np.ndarray:
    ax = math.radians(float(ax_deg))
    ay = math.radians(float(ay_deg))
    n = np.array([math.tan(ax), math.tan(ay), 1.0], dtype=np.float64)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return n / n_norm


def _points_from_depth(
    depth_flat_m: np.ndarray,
    u_flat: np.ndarray,
    v_flat: np.ndarray,
    f: float,
    cx: float,
    cy: float,
    bias: float,
) -> np.ndarray:
    d = depth_flat_m - float(bias)
    x = (u_flat - float(cx)) / float(f)
    y = (v_flat - float(cy)) / float(f)
    dirs = np.stack([x, y, np.ones_like(x)], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs * d[:, None]


def _residuals(
    params: np.ndarray,
    depth_flat_m: np.ndarray,
    u_flat: np.ndarray,
    v_flat: np.ndarray,
    cx_fixed: float,
    cy_fixed: float,
    plane_distance_m: float,
    bias_fixed_m: float,
) -> np.ndarray:
    f, ax_deg, ay_deg = [float(v) for v in params]
    pts = _points_from_depth(depth_flat_m, u_flat, v_flat, f, cx_fixed, cy_fixed, bias_fixed_m)
    n = _plane_normal_from_angles(ax_deg, ay_deg)
    return pts @ n - float(plane_distance_m)


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(x))))


# ---------------------------------------------------------------------------
# 可视化（matplotlib 渲染左侧两图）
# ---------------------------------------------------------------------------
def _make_plane_mesh(
    pts: np.ndarray,
    n: np.ndarray,
    d: float,
    scale_pad: float = 0.05,
    min_half_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = n * float(d)
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(helper, n))) > 0.95:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    e1 = np.cross(n, helper)
    e1 /= max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(n, e1)
    e2 /= max(float(np.linalg.norm(e2)), 1e-12)

    rel = pts - p0[None, :]
    a = rel @ e1
    b = rel @ e2
    a_min, a_max = float(np.min(a)), float(np.max(a))
    b_min, b_max = float(np.min(b)), float(np.max(b))
    a_pad = max((a_max - a_min) * scale_pad, min_half_size)
    b_pad = max((b_max - b_min) * scale_pad, min_half_size)
    a_lin = np.linspace(a_min - a_pad, a_max + a_pad, 20)
    b_lin = np.linspace(b_min - b_pad, b_max + b_pad, 20)

    aa, bb = np.meshgrid(a_lin, b_lin)
    xyz = (
        p0[None, None, :]
        + aa[..., None] * e1[None, None, :]
        + bb[..., None] * e2[None, None, :]
    )
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]


def _draw_3d_plot(ax: Any, points: np.ndarray, residuals: np.ndarray, normal: np.ndarray) -> None:
    px, py, pz = _make_plane_mesh(points, normal, PLANE_DISTANCE_M)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=residuals, cmap="coolwarm", s=22, alpha=0.9)
    ax.plot_surface(px, py, pz, alpha=0.35, color="tab:green", linewidth=0, antialiased=True)
    ax.scatter([0.0], [0.0], [0.0], c="k", s=60, marker="x")
    # labelpad 让放大后的轴标签不和刻度撞在一起。
    ax.set_xlabel("X (m)", labelpad=14)
    ax.set_ylabel("Y (m)", labelpad=14)
    ax.set_zlabel("Z (m)", labelpad=14)
    ax.set_title("ToF 点云 + 拟合平面", pad=18)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _draw_error_distribution_hist(ax: Any, residuals_m: np.ndarray) -> None:
    errs_cm = np.asarray(residuals_m, dtype=np.float64).reshape(-1) * 100.0
    errs_cm = errs_cm[np.isfinite(errs_cm)]
    if errs_cm.size == 0:
        return

    ax.hist(errs_cm, bins=30, color=_HIST_COLOR, edgecolor="white", alpha=0.9)
    ax.set_title("误差分布", pad=14)
    ax.set_xlabel("误差 (cm)", labelpad=10)
    ax.set_ylabel("数量", labelpad=10)
    ax.grid(alpha=0.25, linestyle="--")
    rms_cm = float(np.sqrt(np.mean(errs_cm * errs_cm)))
    ax.text(
        0.02, 0.98,
        f"样本={errs_cm.size}, RMS={rms_cm:.3f} cm",
        transform=ax.transAxes,
        va="top", ha="left",
    )


def _draw_brightness_hist(ax: Any, brightness_map: np.ndarray) -> None:
    """亮度分布直方图：标注 mean / min / max。"""
    vals = np.asarray(brightness_map, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    ax.hist(vals, bins=30, color=_HIST_COLOR, edgecolor="white", alpha=0.9)
    ax.set_title("亮度分布", pad=14)
    ax.set_xlabel("亮度", labelpad=10)
    ax.set_ylabel("数量", labelpad=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.text(
        0.02, 0.98,
        f"均值 = {float(vals.mean()):.1f}\n"
        f"最小 = {float(vals.min()):.1f}\n"
        f"最大 = {float(vals.max()):.1f}",
        transform=ax.transAxes,
        va="top", ha="left",
    )


def _draw_center_pixel_hist(ax: Any, tof_cube: np.ndarray) -> None:
    """画 ToF 中心像素前 62 个 bin 的直方图。"""
    cy = IMG_H // 2
    cx = IMG_W // 2
    bins = np.asarray(tof_cube[cy, cx, :TOF_HIST_VALID_BINS], dtype=np.float64)
    idx = np.arange(bins.size)
    peak_bin = int(np.argmax(bins)) if bins.size > 0 else -1

    ax.bar(idx, bins, width=1.0, color=_HIST_COLOR, edgecolor="white", linewidth=0.4)
    if peak_bin >= 0:
        ax.axvline(peak_bin, color="red", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.set_title(f"中心像素 ({cy}, {cx}) 直方图", pad=14)
    ax.set_xlabel("bin 编号 (0~61)", labelpad=10)
    ax.set_ylabel("数量", labelpad=10)
    ax.grid(alpha=0.25, linestyle="--")
    if peak_bin >= 0:
        ax.text(
            0.98, 0.98,
            f"峰值 bin = {peak_bin}\n峰值数 = {float(bins[peak_bin]):.1f}",
            transform=ax.transAxes,
            va="top", ha="right",
        )


def _draw_brightness_image(ax: Any, brightness_map: np.ndarray) -> None:
    """显示 30x40 亮度图：vmax 直接取数据最大值,自适应配色。"""
    arr = np.asarray(brightness_map, dtype=np.float64)
    vmax = float(np.nanmax(arr)) if arr.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    im = ax.imshow(
        arr,
        cmap="gray",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title("图像亮度", pad=14)
    ax.set_xlabel("列", labelpad=10)
    ax.set_ylabel("行", labelpad=10)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=_CBAR_TICK_FONTSIZE)


def _draw_residual_image(ax: Any, residuals_m: np.ndarray) -> None:
    """把每像素残差画成 30x40 热图（带符号 cm）。"""
    res_cm = np.asarray(residuals_m, dtype=np.float64).reshape(IMG_H, IMG_W) * 100.0
    vmax = float(max(np.max(np.abs(res_cm)), 1e-6))
    im = ax.imshow(
        res_cm,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title("单像素残差 (cm)", pad=14)
    ax.set_xlabel("列", labelpad=10)
    ax.set_ylabel("行", labelpad=10)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=_CBAR_TICK_FONTSIZE)


def _fig_to_rgb_image(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return np.asarray(buf[:, :, :3], dtype=np.uint8)


# 三个直方图（亮度分布 / 误差分布 / 中心像素 hist）共用一个颜色,
# 视觉上更统一,也便于一眼把三张 hist 与下方矩阵图区分开。
_HIST_COLOR = "steelblue"


# 中文字体优先级：Windows 微软雅黑 → 黑体 → 跨平台兜底。
# 任意一个没装也能 fallback 到下一个,负号用 ASCII '-' 避免显示方块。
_CN_FONT_FAMILY = [
    "Microsoft YaHei",
    "SimHei",
    "Microsoft JhengHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "PingFang SC",
    "Arial",
    "DejaVu Sans",
]


# matplotlib 默认字体偏小（10pt），左侧 figure 又会被压缩到面板高度,
# 字体显得太细。中文字体本身笔画就重,不再额外 bold,避免糊在一起。
_PLOT_RC = {
    "font.family":         "sans-serif",
    "font.sans-serif":     _CN_FONT_FAMILY,
    "axes.unicode_minus":  False,
    "font.size":           26,
    "axes.titlesize":      32,
    "axes.labelsize":      26,
    "xtick.labelsize":     22,
    "ytick.labelsize":     22,
    "legend.fontsize":     24,
    "axes.titleweight":    "normal",
    "axes.labelweight":    "normal",
    "axes.linewidth":      1.4,
    "xtick.major.width":   1.2,
    "ytick.major.width":   1.2,
}

# 颜色条刻度字号；与 tick 字号大致一致,但稍小,避免占用 figure 空间。
_CBAR_TICK_FONTSIZE = 20


def _render_visual_left(
    residuals_m: np.ndarray,
    points: np.ndarray,
    normal: np.ndarray,
    brightness_map: np.ndarray,
    tof_cube: np.ndarray,
) -> np.ndarray:
    """渲染 2x3 拼接图（BGR）：

    +---------------------+----------------------+----------------------+
    | (1,1) 亮度直方图     | (1,2) 误差直方图      | (1,3) 中心像素 62-bin |
    +---------------------+----------------------+----------------------+
    | (2,1) 亮度矩阵图     | (2,2) 误差矩阵图      | (2,3) 3D 点云 + 平面  |
    +---------------------+----------------------+----------------------+
    """
    with plt.rc_context(_PLOT_RC):
        fig = plt.figure(figsize=(18 * VISUAL_RES_SCALE, 12 * VISUAL_RES_SCALE))

        ax_bhist = fig.add_subplot(2, 3, 1)
        _draw_brightness_hist(ax_bhist, brightness_map)

        ax_ehist = fig.add_subplot(2, 3, 2)
        _draw_error_distribution_hist(ax_ehist, residuals_m)

        ax_chist = fig.add_subplot(2, 3, 3)
        _draw_center_pixel_hist(ax_chist, tof_cube)

        ax_bimg = fig.add_subplot(2, 3, 4)
        _draw_brightness_image(ax_bimg, brightness_map)

        ax_resid = fig.add_subplot(2, 3, 5)
        _draw_residual_image(ax_resid, residuals_m)

        ax_3d = fig.add_subplot(2, 3, 6, projection="3d")
        _draw_3d_plot(ax_3d, points, residuals_m, normal)

        fig.tight_layout()
        rgb = _fig_to_rgb_image(fig)
        plt.close(fig)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# 阈值检查
# ---------------------------------------------------------------------------
def _metric_in_range(value: float, cfg: dict[str, float]) -> tuple[bool, float, float]:
    if "min" not in cfg or "max" not in cfg:
        raise ValueError("threshold config must contain both 'min' and 'max'")
    min_v = float(cfg["min"])
    max_v = float(cfg["max"])
    return (min_v <= value <= max_v), min_v, max_v


def _mk_item(name: str, status: str, measured: str, threshold: str, note: str = "") -> dict[str, str]:
    return {
        "name": name,
        "status": status,
        "measured": measured,
        "threshold": threshold,
        "note": note,
    }


_METRIC_DISPLAY: dict[str, tuple[str, str, str]] = {
    # name -> (中文名, 单位, 数值格式)
    "f":         ("焦距 f",        "px",  "{:.3f}"),
    "bias":      ("距离偏置 bias", "cm",  "{:+.2f}"),
    "ax":        ("X 倾角 ax",     "deg", "{:+.3f}"),
    "ay":        ("Y 倾角 ay",     "deg", "{:+.3f}"),
    "rms":       ("RMS 误差",      "cm",  "{:.3f}"),
    "worst":     ("最坏 1% 误差",  "cm",  "{:.3f}"),
    "peak_mean": ("亮度均值",      "",    "{:.1f}"),
    "peak_max":  ("亮度最大值",    "",    "{:.1f}"),
    "peak_min":  ("亮度最小值",    "",    "{:.1f}"),
}

_SECTIONS_LAYOUT: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("几何标定", ("f", "bias", "ax", "ay")),
    ("平面误差", ("rms", "worst")),
    ("光度统计", ("peak_mean", "peak_max", "peak_min")),
)


def _build_sections(values: dict[str, float]) -> tuple[list[tuple[str, list[dict[str, str]]]], bool]:
    sections: list[tuple[str, list[dict[str, str]]]] = []
    overall_pass = True
    for sec_title, names in _SECTIONS_LAYOUT:
        items: list[dict[str, str]] = []
        for name in names:
            cfg = _THRESHOLDS.get(name)
            if cfg is None:
                items.append(_mk_item(name, "SKIP", "-", "-", "无阈值配置"))
                continue
            value = float(values[name])
            ok, lo, hi = _metric_in_range(value, cfg)
            cn_name, unit, fmt = _METRIC_DISPLAY.get(name, (name, "", "{:.4f}"))
            measured = fmt.format(value) + (f" {unit}" if unit else "")
            threshold = f"[{fmt.format(lo)}, {fmt.format(hi)}]"
            note = "" if ok else "超出范围"
            items.append(_mk_item(cn_name, "PASS" if ok else "FAIL", measured, threshold, note))
            if not ok:
                overall_pass = False
        sections.append((sec_title, items))
    return sections, overall_pass


# ---------------------------------------------------------------------------
# 右侧产测项目面板（中文，PIL 绘制）
# ---------------------------------------------------------------------------
_STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "PASS": (60, 200, 60),    # green (BGR)
    "FAIL": (60, 80, 230),    # red
    "SKIP": (160, 160, 160),  # gray
}
_STATUS_TEXT_CN = {"PASS": "通过", "FAIL": "失败", "SKIP": "跳过"}

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    r"/System/Library/Fonts/PingFang.ttc",
)
_FONT_BOLD_CANDIDATES = (
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
)
_FONT_PATH = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
_FONT_PATH_BOLD = next((p for p in _FONT_BOLD_CANDIDATES if os.path.exists(p)), _FONT_PATH)
_FONT_CACHE: dict[tuple[int, bool], Any] = {}


def _get_font(size: int, bold: bool = False) -> Any:
    key = (size, bool(bold))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = _FONT_PATH_BOLD if bold else _FONT_PATH
    if path is None:
        font = ImageFont.load_default()
    else:
        try:
            font = ImageFont.truetype(path, size)
        except Exception:
            font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    size: int = 16,
    bold: bool = False,
    align: str = "left",
) -> None:
    font = _get_font(size, bold=bold)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x, y = org
    if align == "right":
        x = x - tw
    elif align == "center":
        x = x - tw // 2
    y_top = int(y) - th
    draw.text(
        (int(x), y_top), text, font=font,
        fill=(int(color[2]), int(color[1]), int(color[0])),
    )
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _draw_test_panel(
    sections: list[tuple[str, list[dict[str, str]]]],
    overall_pass: bool,
    panel_w: int,
) -> np.ndarray:
    pad_x = 18
    title_h = 56
    line_h = 30
    section_head_h = 34
    section_gap = 10
    bottom_pad = 16

    n_items = sum(len(its) for _, its in sections)
    panel_h = (
        title_h
        + len(sections) * (section_head_h + section_gap)
        + n_items * line_h
        + bottom_pad
    )
    panel = np.full((panel_h, panel_w, 3), 24, dtype=np.uint8)

    title_y = 38
    _put_text(panel, "产测项目", (pad_x, title_y), (235, 235, 235), size=24, bold=True)
    overall_text = f"总判定: {'通过' if overall_pass else '失败'}"
    overall_color = _STATUS_COLORS["PASS" if overall_pass else "FAIL"]
    _put_text(
        panel, overall_text, (panel_w - pad_x, title_y),
        overall_color, size=24, bold=True, align="right",
    )
    cv2.line(panel, (pad_x, title_h - 4), (panel_w - pad_x, title_h - 4), (110, 110, 110), 1)

    col_name_x = pad_x
    col_measured_right = int(panel_w * 0.50)
    col_threshold_x = col_measured_right + 12
    col_status_right = panel_w - pad_x

    section_color = (170, 200, 255)
    y = title_h + section_gap

    for sec_title, items in sections:
        head_baseline = y + section_head_h - 8
        bar_top = y + 6
        bar_bottom = y + section_head_h - 4
        cv2.rectangle(panel, (pad_x - 6, bar_top), (pad_x - 2, bar_bottom), section_color, -1)
        _put_text(panel, sec_title, (pad_x + 4, head_baseline), section_color, size=20, bold=True)
        sub_sep_y = y + section_head_h - 2
        cv2.line(panel, (pad_x, sub_sep_y), (panel_w - pad_x, sub_sep_y), (70, 70, 90), 1)
        y += section_head_h + 4

        for it in items:
            color = _STATUS_COLORS.get(it["status"], (220, 220, 220))
            status_cn = _STATUS_TEXT_CN.get(it["status"], it["status"])
            _put_text(panel, it["name"], (col_name_x, y + line_h - 8), color, size=16)
            _put_text(
                panel, it["measured"], (col_measured_right, y + line_h - 8),
                color, size=16, align="right",
            )
            _put_text(
                panel, it["threshold"], (col_threshold_x, y + line_h - 8),
                (210, 210, 210), size=15,
            )
            _put_text(
                panel, f"[{status_cn}]", (col_status_right, y + line_h - 8),
                color, size=17, bold=True, align="right",
            )
            y += line_h

        y += section_gap

    return panel


# ---------------------------------------------------------------------------
# 拼接：左 visual + 右 panel
# ---------------------------------------------------------------------------
_OUTPUT_WIDTH = 1800
_PANEL_WIDTH = 560
_PANEL_SEP_WIDTH = 2
_LEFT_WIDTH = _OUTPUT_WIDTH - _PANEL_WIDTH - _PANEL_SEP_WIDTH
_HEADER_HEIGHT = 50


def _compose_combined_image(
    visual_left_bgr: np.ndarray,
    sections: list[tuple[str, list[dict[str, str]]]],
    overall_pass: bool,
) -> np.ndarray:
    src = visual_left_bgr
    sh, sw = int(src.shape[0]), int(src.shape[1])
    target_w = _LEFT_WIDTH
    body_h = max(int(round(sh * target_w / max(sw, 1))), 1)
    body = cv2.resize(src, (target_w, body_h), interpolation=cv2.INTER_AREA)

    header = np.zeros((_HEADER_HEIGHT, target_w, 3), dtype=np.uint8)
    _put_text(
        header,
        "ToF 标定可视化（亮度分布 / 误差分布 / 中心像素 hist / 亮度图 / 残差图 / 3D 点云）",
        (12, 34), (255, 255, 255), size=22, bold=True,
    )
    left = np.vstack([header, body])

    panel = _draw_test_panel(sections, overall_pass, _PANEL_WIDTH)

    out_h = max(left.shape[0], panel.shape[0])
    if left.shape[0] < out_h:
        pad = np.zeros((out_h - left.shape[0], left.shape[1], 3), dtype=np.uint8)
        left = np.vstack([left, pad])
    if panel.shape[0] < out_h:
        pad = np.full((out_h - panel.shape[0], panel.shape[1], 3), 24, dtype=np.uint8)
        panel = np.vstack([panel, pad])

    sep = np.full((out_h, _PANEL_SEP_WIDTH, 3), 70, dtype=np.uint8)
    return np.hstack([left, sep, panel])


# ---------------------------------------------------------------------------
# 标定主流程
# ---------------------------------------------------------------------------
def _calibrate(tof_cube: np.ndarray) -> dict[str, Any]:
    depth_map = _depth_from_hist_centroid(tof_cube)
    peak_brightness = _compute_peak_brightness(tof_cube)
    bias_fixed = _compute_bias_from_depth(depth_map, PLANE_DISTANCE_M)
    depth_flat = depth_map.reshape(-1)
    u_flat, v_flat = _build_roi_uv()

    cx0 = (IMG_W - 1) / 2.0
    cy0 = (IMG_H - 1) / 2.0

    x0 = np.array([F_INIT, AX_INIT_DEG, AY_INIT_DEG], dtype=np.float64)
    bounds = [(F_MIN, F_MAX), (AX_MIN_DEG, AX_MAX_DEG), (AY_MIN_DEG, AY_MAX_DEG)]

    def objective_rms(p: np.ndarray) -> float:
        r = _residuals(p, depth_flat, u_flat, v_flat, cx0, cy0, PLANE_DISTANCE_M, bias_fixed)
        return _rms(r)

    pw_res = minimize(
        objective_rms,
        x0=np.asarray(x0, dtype=np.float64),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": POWELL_MAXITER,
            "xtol": POWELL_XTOL,
            "ftol": POWELL_FTOL,
            "disp": False,
        },
    )

    x_opt = np.asarray(pw_res.x, dtype=np.float64)
    r_opt = _residuals(x_opt, depth_flat, u_flat, v_flat, cx0, cy0, PLANE_DISTANCE_M, bias_fixed)
    rms_m = _rms(r_opt)

    f, ax_deg, ay_deg = [float(v) for v in x_opt]
    bias = float(bias_fixed)
    normal = _plane_normal_from_angles(ax_deg, ay_deg)

    pts_opt = _points_from_depth(depth_flat, u_flat, v_flat, f, cx0, cy0, bias)

    abs_err = np.abs(r_opt)
    worst_k = max(1, int(math.ceil(abs_err.size * WORST_ERROR_TOP_RATIO)))
    worst_top_threshold_m = float(
        np.partition(abs_err, abs_err.size - worst_k)[abs_err.size - worst_k]
    )

    pb_flat = np.asarray(peak_brightness, dtype=np.float64).reshape(-1)
    pb_flat = pb_flat[np.isfinite(pb_flat)]
    pb_mean = float(np.mean(pb_flat)) if pb_flat.size > 0 else 0.0
    pb_max = float(np.max(pb_flat)) if pb_flat.size > 0 else 0.0
    pb_min = float(np.min(pb_flat)) if pb_flat.size > 0 else 0.0

    # 误差 / 偏置统一以 cm 对外暴露。
    return {
        "values": {
            "f": f,
            "bias": bias * 100.0,
            "ax": ax_deg,
            "ay": ay_deg,
            "rms": rms_m * 100.0,
            "worst": worst_top_threshold_m * 100.0,
            "peak_mean": pb_mean,
            "peak_max": pb_max,
            "peak_min": pb_min,
        },
        "residuals": r_opt,
        "points": pts_opt,
        "normal": normal,
        "brightness_map": np.asarray(peak_brightness, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def run_all_checks(tof_raw_path: str) -> tuple[bool, np.ndarray, list[float]]:
    """对一帧 ToF raw 做完整产测。

    参数
    ----
    tof_raw_path : str
        ToF raw 文件路径，绝对/相对都行（相对路径基于调用时 cwd）。

    返回
    ----
    tuple ``(passed, image, params)``:
        passed : bool
            所有 metric 同时落在阈值范围内。
        image : numpy.ndarray
            BGR 图像，左侧为 2x3 标定可视化：
            行 1 为三个直方图（亮度分布 / 误差分布 / 中心像素 62-bin），
            行 2 为亮度矩阵图 / 残差矩阵图 / 3D 点云 + 拟合平面；
            右侧为产测项目面板（按"几何标定 / 平面误差 / 光度统计"分组，
            每一项都列出 measured / threshold / 状态）。
        params : list[float]
            9 个 metric 数值，顺序固定为
            ``[f(px), bias(cm), ax(deg), ay(deg),
               rms(cm), worst(cm), peak_mean, peak_max, peak_min]``。
    """
    original_cwd = os.getcwd()
    abs_raw_path = (
        tof_raw_path if os.path.isabs(tof_raw_path)
        else os.path.abspath(os.path.join(original_cwd, tof_raw_path))
    )
    if not os.path.isfile(abs_raw_path):
        raise FileNotFoundError(f"找不到输入raw: {abs_raw_path}")

    os.makedirs(_TMP_DIR, exist_ok=True)

    tof_cube = _load_tof_raw_cube(abs_raw_path)
    cali = _calibrate(tof_cube)

    values = cali["values"]
    sections, overall_pass = _build_sections(values)

    visual_left_bgr = _render_visual_left(
        cali["residuals"], cali["points"], cali["normal"],
        cali["brightness_map"], tof_cube,
    )
    image = _compose_combined_image(visual_left_bgr, sections, overall_pass)

    params: list[float] = [float(values[name]) for name in _METRIC_NAMES]
    return bool(overall_pass), image, params

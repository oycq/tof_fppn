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

# 左侧 figure 渲染参数:
#   - figsize 始终按 (target_w / _VISUAL_DPI, target_h / _VISUAL_DPI) 算,
#     这样字号 / 坐标轴线宽相对目标尺寸是固定比例,不会变形。
#   - 内部用 dpi = _VISUAL_DPI * _VISUAL_OVERSAMPLE 渲染,得到 2x 像素的
#     超采样图,然后 INTER_AREA 下采样到目标尺寸。这一步是 supersample
#     抗锯齿:对 3D 点云、细线条特别有效,字体也会更锐利。
_VISUAL_DPI = 100
_VISUAL_OVERSAMPLE = 2

_METRIC_NAMES: tuple[str, ...] = (
    "f", "ax", "ay",
    "bias",
    "rms", "worst",
    "dead_pixels",
    "crosstalk_max", "crosstalk_mean",
    "noise_max", "noise_mean",
    "light_max", "light_mean",
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


def _compute_compensated_cube(tof_cube: np.ndarray) -> np.ndarray:
    """对前 62 个 bin 做最后两个 bin 的饱和补偿,返回 (H, W, 62) float64。

    校正公式与 _compute_peak_brightness 一致：
        bin_corr[i, j, k] = bin[i, j, k] * SAT_SCALE / (bin[i,j,62]*1024 + bin[i,j,63])
    分母为 0 时整像素的补偿值置 0（与原始即为 0 等价,不影响后续判定）。
    """
    h = np.asarray(tof_cube, dtype=np.float64)
    denom = h[:, :, 62] * SAT_HIGH_BIN_WEIGHT + h[:, :, 63]
    sat_coeff = np.where(denom > 1e-12, SAT_SCALE / denom, 0.0)
    return h[:, :, :TOF_HIST_VALID_BINS] * sat_coeff[:, :, None]


# 串光统计窗口固定为 bin[0]；底噪窗口按用户口径取 bin[30:50]。
NOISE_BIN_LO = 30
NOISE_BIN_HI = 50  # 不含


def _topk_pixel_positions(score_map: np.ndarray, k: int) -> list[tuple[int, int]]:
    """返回 ``score_map`` 中得分最高的 k 个像素位置 (row, col),按降序。"""
    arr = np.asarray(score_map, dtype=np.float64)
    flat = arr.reshape(-1)
    n = flat.size
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    idx = np.argpartition(flat, n - k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    w = arr.shape[1] if arr.ndim >= 2 else 1
    return [(int(i // w), int(i % w)) for i in idx]


def _compute_extra_metrics(tof_cube: np.ndarray) -> dict[str, Any]:
    """从 raw cube 抽出新增的 7 个产测量。

    所有 max / mean 类指标都基于 *补偿后* 的 hist。
    返回 dict 同时携带绘图所需的中间产物 (含完整 compensated cube,
    以及串扰 / 底噪 top-2 像素位置用于第 3 行单像素直方图)。
    """
    h = np.asarray(tof_cube, dtype=np.float64)
    raw62 = h[:, :, :TOF_HIST_VALID_BINS]

    # 坏点：前 62 bin 全为 0 的像素（用原始值判断,补偿前后等价）。
    dead_mask = np.all(raw62 == 0.0, axis=2)
    dead_count = int(np.sum(dead_mask))

    comp = _compute_compensated_cube(tof_cube)

    bin0 = comp[:, :, 0]
    crosstalk_max = float(np.max(bin0)) if bin0.size else 0.0
    crosstalk_mean = float(np.mean(bin0)) if bin0.size else 0.0

    noise_block = comp[:, :, NOISE_BIN_LO:NOISE_BIN_HI]
    # 底噪以"每像素 bin[NOISE_LO:NOISE_HI] 均值"为基本单位,再对所有像素
    # 取 max/mean,与"打光强度"按像素 peak 统计的口径保持一致。
    noise_per_pixel = (
        np.mean(noise_block, axis=2) if noise_block.size else np.zeros_like(bin0)
    )
    noise_max = float(np.max(noise_per_pixel)) if noise_per_pixel.size else 0.0
    noise_mean = float(np.mean(noise_per_pixel)) if noise_per_pixel.size else 0.0

    peak_per_pixel = np.max(comp, axis=2) if comp.size else np.zeros((IMG_H, IMG_W))
    light_max = float(np.max(peak_per_pixel)) if peak_per_pixel.size else 0.0
    light_mean = float(np.mean(peak_per_pixel)) if peak_per_pixel.size else 0.0

    # 串光 top-2: 按 bin[0] 排序;底噪 top-2: 按"每像素 bin[NOISE_LO:NOISE_HI] 均值"排序。
    crosstalk_top2 = _topk_pixel_positions(bin0, k=2)
    noise_top2 = _topk_pixel_positions(noise_per_pixel, k=2)

    return {
        "values": {
            "dead_pixels":    dead_count,
            "crosstalk_max":  crosstalk_max,
            "crosstalk_mean": crosstalk_mean,
            "noise_max":      noise_max,
            "noise_mean":     noise_mean,
            "light_max":      light_max,
            "light_mean":     light_mean,
        },
        "bin0_per_pixel":  bin0,
        "peak_per_pixel":  peak_per_pixel,
        "noise_block":     noise_block,           # (H, W, NOISE_BIN_HI - NOISE_BIN_LO)
        "noise_per_pixel": noise_per_pixel,       # (H, W) 每像素 bin[30:50] 均值
        "dead_mask":       dead_mask,
        "comp_cube":       comp,                  # (H, W, 62) 补偿后,供单像素 hist 用
        "crosstalk_top2":  crosstalk_top2,
        "noise_top2":      noise_top2,
    }


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
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=residuals, cmap="coolwarm", s=8, alpha=0.85,
        depthshade=True,
    )
    ax.plot_surface(px, py, pz, alpha=0.30, color="tab:green", linewidth=0, antialiased=True)
    ax.scatter([0.0], [0.0], [0.0], c="k", s=28, marker="x", linewidths=1.5)
    ax.set_xlabel("X (m)", labelpad=4)
    ax.set_ylabel("Y (m)", labelpad=4)
    ax.set_zlabel("Z (m)", labelpad=4)
    ax.set_title("ToF 点云 + 拟合平面", pad=6)
    # 点云在 z 方向几乎是一片薄板,把 z 方向压扁,让 X/Y 维度占满更多
    # 视觉空间,3D 散点不再挤成一团。
    ax.set_box_aspect((1.4, 1.4, 0.9))


def _draw_parallelism_hist(ax: Any, residuals_m: np.ndarray) -> None:
    """平行度分布：拟合平面残差越窄 → 越平行。"""
    errs_cm = np.asarray(residuals_m, dtype=np.float64).reshape(-1) * 100.0
    errs_cm = errs_cm[np.isfinite(errs_cm)]
    if errs_cm.size == 0:
        return

    ax.hist(errs_cm, bins=30, color=_HIST_COLOR, edgecolor="white", alpha=0.9)
    ax.set_title("平行度分布", pad=6)
    ax.set_xlabel("残差 (cm)", labelpad=4)
    ax.set_ylabel("数量", labelpad=4)
    ax.grid(alpha=0.25, linestyle="--")
    rms_cm = float(np.sqrt(np.mean(errs_cm * errs_cm)))
    worst_cm = float(np.max(np.abs(errs_cm)))
    ax.text(
        0.98, 0.98,
        f"RMS  = {rms_cm:.3f} cm\n"
        f"最大 = {worst_cm:.3f} cm",
        transform=ax.transAxes,
        va="top", ha="right",
    )


def _draw_brightness_hist(ax: Any, brightness_map: np.ndarray) -> None:
    """亮度分布直方图：标注 mean / min / max。"""
    vals = np.asarray(brightness_map, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    ax.hist(vals, bins=30, color=_HIST_COLOR, edgecolor="white", alpha=0.9)
    ax.set_title("亮度分布", pad=6)
    ax.set_xlabel("亮度", labelpad=4)
    ax.set_ylabel("数量", labelpad=4)
    ax.grid(alpha=0.25, linestyle="--")
    ax.text(
        0.98, 0.98,
        f"均值 = {float(vals.mean()):.1f}\n"
        f"最小 = {float(vals.min()):.1f}\n"
        f"最大 = {float(vals.max()):.1f}",
        transform=ax.transAxes,
        va="top", ha="right",
    )


def _draw_crosstalk_hist(ax: Any, bin0_per_pixel: np.ndarray) -> None:
    """串光直方图：所有像素的 bin[0] 补偿值分布。"""
    vals = np.asarray(bin0_per_pixel, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    ax.hist(vals, bins=30, color=_HIST_COLOR, edgecolor="white", alpha=0.9)
    ax.set_title("串光分布 (bin[0])", pad=6)
    ax.set_xlabel("bin[0] 补偿值", labelpad=4)
    ax.set_ylabel("数量", labelpad=4)
    ax.grid(alpha=0.25, linestyle="--")
    ax.text(
        0.98, 0.98,
        f"均值 = {float(vals.mean()):.1f}\n"
        f"最大 = {float(vals.max()):.1f}",
        transform=ax.transAxes,
        va="top", ha="right",
    )


def _draw_noise_hist(ax: Any, noise_per_pixel: np.ndarray) -> None:
    """底噪直方图：每像素 bin[NOISE_LO:NOISE_HI] 均值的分布。"""
    vals = np.asarray(noise_per_pixel, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    ax.hist(vals, bins=30, color=_HIST_COLOR, edgecolor="white", alpha=0.9)
    ax.set_title(f"底噪分布 (各像素 bin[{NOISE_BIN_LO}:{NOISE_BIN_HI}] 均值)", pad=6)
    ax.set_xlabel("底噪", labelpad=4)
    ax.set_ylabel("数量", labelpad=4)
    ax.grid(alpha=0.25, linestyle="--")
    ax.text(
        0.98, 0.98,
        f"均值 = {float(vals.mean()):.2f}\n"
        f"最大 = {float(vals.max()):.2f}",
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
    ax.set_title("图像亮度", pad=6)
    ax.set_xlabel("列", labelpad=4)
    ax.set_ylabel("行", labelpad=4)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=_CBAR_TICK_FONTSIZE)


def _draw_crosstalk_image(ax: Any, bin0_per_pixel: np.ndarray) -> None:
    """串光 2D 图：每像素 bin[0] 补偿值。"""
    arr = np.asarray(bin0_per_pixel, dtype=np.float64)
    vmax = float(np.nanmax(arr)) if arr.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    im = ax.imshow(
        arr,
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title("串光分布 (bin[0])", pad=6)
    ax.set_xlabel("列", labelpad=4)
    ax.set_ylabel("行", labelpad=4)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=_CBAR_TICK_FONTSIZE)


def _draw_noise_image(ax: Any, noise_per_pixel: np.ndarray) -> None:
    """底噪 2D 图：每像素 bin[NOISE_LO:NOISE_HI] 均值,凸显底噪偏高的像素。"""
    arr = np.asarray(noise_per_pixel, dtype=np.float64)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    vmax = float(np.nanmax(arr)) if arr.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    im = ax.imshow(
        arr,
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(f"底噪 (bin[{NOISE_BIN_LO}:{NOISE_BIN_HI}] 均值)", pad=6)
    ax.set_xlabel("列", labelpad=4)
    ax.set_ylabel("行", labelpad=4)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=_CBAR_TICK_FONTSIZE)


def _draw_pixel_bin_hist(
    ax: Any,
    bins_62: np.ndarray,
    pos: tuple[int, int],
    title_prefix: str,
    highlight_range: tuple[int, int] | None = None,
    annotation: str | None = None,
) -> None:
    """绘制单个像素的 bin[0:62] 补偿值柱状图。

    ``highlight_range = (lo, hi)`` 用橙红色高亮关注区间(串光高亮 bin[0],
    底噪高亮 bin[NOISE_LO:NOISE_HI]),便于一眼对位。
    ``annotation`` 写在右上角,通常用来标这个像素的 metric 数值,
    比如 "串光 = 222.0" 或 "底噪 = 23.45"。
    """
    vals = np.asarray(bins_62, dtype=np.float64).reshape(-1)
    n = vals.size
    x = np.arange(n)

    ax.bar(x, vals, width=1.0, color=_HIST_COLOR, edgecolor="none")
    if highlight_range is not None and n > 0:
        lo = max(int(highlight_range[0]), 0)
        hi = min(int(highlight_range[1]), n)
        if hi > lo:
            ax.bar(
                x[lo:hi], vals[lo:hi],
                width=1.0, color="orangered", alpha=0.9, edgecolor="none",
            )

    r, c = int(pos[0]), int(pos[1])
    ax.set_title(f"{title_prefix} (r,c)=({r},{c})", pad=4)
    ax.set_xlabel("bin", labelpad=3)
    ax.set_ylabel("补偿值", labelpad=3)
    ax.grid(alpha=0.25, linestyle="--", axis="y")
    ax.set_xlim(-0.5, n - 0.5)
    if annotation:
        ax.text(
            0.98, 0.98, annotation,
            transform=ax.transAxes,
            va="top", ha="right",
        )


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


# 左侧 figure 直接渲染到目标像素 (~1240 x 760, dpi=100),
# 字号按这个分辨率挑成"清晰且不占满子图"的水平。
# 中文字体笔画本身就重,不再 bold,避免糊在一起。
_PLOT_RC = {
    "font.family":         "sans-serif",
    "font.sans-serif":     _CN_FONT_FAMILY,
    "axes.unicode_minus":  False,
    "font.size":           8,
    "axes.titlesize":      10,
    "axes.labelsize":      8,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "legend.fontsize":     7,
    "axes.titleweight":    "normal",
    "axes.labelweight":    "normal",
    "axes.linewidth":      0.8,
    "xtick.major.width":   0.7,
    "ytick.major.width":   0.7,
}

# 颜色条刻度字号；与 tick 字号匹配。
_CBAR_TICK_FONTSIZE = 7


def _render_visual_left(
    residuals_m: np.ndarray,
    points: np.ndarray,
    normal: np.ndarray,
    brightness_map: np.ndarray,
    bin0_per_pixel: np.ndarray,
    noise_per_pixel: np.ndarray,
    comp_cube: np.ndarray,
    crosstalk_top2: list[tuple[int, int]],
    noise_top2: list[tuple[int, int]],
    target_size: tuple[int, int],
) -> np.ndarray:
    """渲染 3x4 拼接图（BGR）。

    figure 直接按 ``target_size = (target_w, target_h)`` 像素绘制,
    后续不再做 ``cv2.resize`` 缩放,所以字体/坐标轴不会被二次拉伸,
    保证清晰、不变形。

    布局:

    +-------------+-------------+-------------+-------------+
    | (1,1) 亮度  | (1,2) 串光  | (1,3) 底噪  | (1,4) 平行度|
    |     直方图  |     直方图  |     直方图  |     直方图  |
    +-------------+-------------+-------------+-------------+
    | (2,1) 亮度  | (2,2) 串光  | (2,3) 底噪  | (2,4) 3D    |
    |     2D 图   |     2D 图   |     2D 图   |     点云    |
    +-------------+-------------+-------------+-------------+
    | (3,1) 串光  | (3,2) 串光  | (3,3) 底噪  | (3,4) 底噪  |
    |  最差像素#1 |  最差像素#2 |  最大像素#1 |  最大像素#2 |
    |   bin[0:62] |   bin[0:62] |   bin[0:62] |   bin[0:62] |
    +-------------+-------------+-------------+-------------+
    """
    target_w, target_h = int(target_size[0]), int(target_size[1])
    target_w = max(target_w, 1)
    target_h = max(target_h, 1)

    with plt.rc_context(_PLOT_RC):
        # figsize 按"目标像素 / 基准 dpi"算,字号相对画面比例 = 设定值。
        # 实际渲染 dpi 加倍 (oversample),让 3D 散点 / 细线条以 2x 精度绘制。
        fig = plt.figure(
            figsize=(target_w / _VISUAL_DPI, target_h / _VISUAL_DPI),
            dpi=_VISUAL_DPI * _VISUAL_OVERSAMPLE,
        )
        # 第二行 (2D / 3D) 细节最多,留最大;第一/三行直方图信息密度低,稍短。
        gs = fig.add_gridspec(3, 4, height_ratios=[0.95, 1.20, 0.90])

        # row 1 — 直方图
        _draw_brightness_hist(fig.add_subplot(gs[0, 0]), brightness_map)
        _draw_crosstalk_hist(fig.add_subplot(gs[0, 1]),  bin0_per_pixel)
        _draw_noise_hist(fig.add_subplot(gs[0, 2]),      noise_per_pixel)
        _draw_parallelism_hist(fig.add_subplot(gs[0, 3]), residuals_m)

        # row 2 — 2D / 3D 图
        _draw_brightness_image(fig.add_subplot(gs[1, 0]), brightness_map)
        _draw_crosstalk_image(fig.add_subplot(gs[1, 1]),  bin0_per_pixel)
        _draw_noise_image(fig.add_subplot(gs[1, 2]),      noise_per_pixel)
        ax_3d = fig.add_subplot(gs[1, 3], projection="3d")
        _draw_3d_plot(ax_3d, points, residuals_m, normal)

        # row 3 — 串光 / 底噪 最差像素的 bin[0:62] 直方图
        # (3,1)/(3,2): 串光 top-2 (高亮 bin[0],右上角标这个像素的串光值)
        # (3,3)/(3,4): 底噪 top-2 (高亮 bin[NOISE_LO:NOISE_HI],右上角标底噪值)
        row3_slots: list[dict[str, Any] | None] = []
        for i in range(2):
            if i < len(crosstalk_top2):
                r, c = crosstalk_top2[i]
                row3_slots.append({
                    "pos":        (r, c),
                    "title":      f"串光最差 #{i + 1}",
                    "highlight":  (0, 1),
                    "annotation": f"串光 = {float(bin0_per_pixel[r, c]):.1f}",
                })
            else:
                row3_slots.append(None)
        for i in range(2):
            if i < len(noise_top2):
                r, c = noise_top2[i]
                row3_slots.append({
                    "pos":        (r, c),
                    "title":      f"底噪最大 #{i + 1}",
                    "highlight":  (NOISE_BIN_LO, NOISE_BIN_HI),
                    "annotation": f"底噪 = {float(noise_per_pixel[r, c]):.2f}",
                })
            else:
                row3_slots.append(None)

        for col, spec in enumerate(row3_slots):
            ax = fig.add_subplot(gs[2, col])
            if spec is None:
                ax.axis("off")
                continue
            r, c = spec["pos"]
            _draw_pixel_bin_hist(
                ax, comp_cube[r, c, :], (r, c),
                spec["title"],
                highlight_range=spec["highlight"],
                annotation=spec["annotation"],
            )

        # 子图间距收紧,让格子之间的留白尽量小,作图区相对更大。
        fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.4)
        rgb = _fig_to_rgb_image(fig)
        plt.close(fig)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # supersample 之后用 INTER_AREA 下采样到目标尺寸,等价于盒式平均抗锯齿,
    # 文字 / 3D 点云都比直接小 dpi 渲染更锐利。
    if bgr.shape[:2] != (target_h, target_w):
        bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return bgr


# ---------------------------------------------------------------------------
# 阈值检查
# ---------------------------------------------------------------------------
def _metric_in_range(
    value: float, cfg: dict[str, float]
) -> tuple[bool, float | None, float | None]:
    """支持三种阈值形式：

    - ``{"max": X}``           → 仅上限,要求 ``value <= X``
    - ``{"min": Y}``           → 仅下限,要求 ``value >= Y``
    - ``{"min": Y, "max": X}`` → 区间,要求 ``Y <= value <= X``

    返回 ``(passed, min_v_or_None, max_v_or_None)``,便于上层按需展示。
    """
    has_min = "min" in cfg
    has_max = "max" in cfg
    if not has_min and not has_max:
        raise ValueError("threshold config must contain at least one of 'min' / 'max'")

    min_v = float(cfg["min"]) if has_min else None
    max_v = float(cfg["max"]) if has_max else None

    ok = True
    if min_v is not None and value < min_v:
        ok = False
    if max_v is not None and value > max_v:
        ok = False
    return ok, min_v, max_v


def _format_threshold(min_v: float | None, max_v: float | None, fmt: str) -> str:
    if min_v is not None and max_v is not None:
        return f"[{fmt.format(min_v)}, {fmt.format(max_v)}]"
    if max_v is not None:
        return f"<= {fmt.format(max_v)}"
    if min_v is not None:
        return f">= {fmt.format(min_v)}"
    return "-"


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
    "f":              ("焦距 f",        "px",  "{:.3f}"),
    "bias":           ("FPPN 偏置",     "cm",  "{:+.2f}"),
    "ax":             ("X 倾角 ax",     "deg", "{:+.3f}"),
    "ay":             ("Y 倾角 ay",     "deg", "{:+.3f}"),
    "rms":            ("均值 (RMS)",    "cm",  "{:.3f}"),
    "worst":          ("最大 (1%)",     "cm",  "{:.3f}"),
    "dead_pixels":    ("坏点数量",      "",    "{:.0f}"),
    "crosstalk_max":  ("最大值",        "",    "{:.1f}"),
    "crosstalk_mean": ("均值",          "",    "{:.1f}"),
    "noise_max":      ("最大值",        "",    "{:.1f}"),
    "noise_mean":     ("均值",          "",    "{:.1f}"),
    "light_max":      ("最大值",        "",    "{:.1f}"),
    "light_mean":     ("均值",          "",    "{:.1f}"),
}

_SECTIONS_LAYOUT: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("坏点检测",   ("dead_pixels",)),
    ("串光检测",   ("crosstalk_max", "crosstalk_mean")),
    ("底噪检测",   ("noise_max", "noise_mean")),
    ("打光强度",   ("light_max", "light_mean")),
    ("几何标定",   ("f", "ax", "ay")),
    ("FPPN 检测", ("bias",)),
    ("平面度",     ("rms", "worst")),
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
            threshold = _format_threshold(lo, hi, fmt)
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

    # 与下方 y 累加同步：起点 title_h + section_gap，
    # 每个 section 头占 section_head_h + 4，section 末尾再 += section_gap，
    # 每个 item 行占 line_h。
    n_items = sum(len(its) for _, its in sections)
    panel_h = (
        title_h + section_gap
        + len(sections) * (section_head_h + 4 + section_gap)
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
    # measured 列稍微左移,留更多宽度给 threshold 字段(如 "[50.000, 59.000]"),
    # 这样 panel 整体可以缩窄而不截字。
    col_measured_right = int(panel_w * 0.46)
    col_threshold_x = col_measured_right + 10
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
_PANEL_WIDTH = 500
_PANEL_SEP_WIDTH = 2
_LEFT_WIDTH = _OUTPUT_WIDTH - _PANEL_WIDTH - _PANEL_SEP_WIDTH
_HEADER_HEIGHT = 50

# 左侧 figure(不含 header)的最小像素高度。3x4 布局一行 ~270px 即可清晰,
# 整体看起来更扁更协调;panel 比此短时下方补暗灰背景,长则左侧补黑。
_MIN_LEFT_BODY_H = 820


def _compose_combined_image(
    visual_left_bgr: np.ndarray,
    panel: np.ndarray,
) -> np.ndarray:
    """把"已经按目标像素绘好"的左侧 figure 与右侧 panel 拼起来。

    入参约定:
      - ``visual_left_bgr.shape[1] == _LEFT_WIDTH``
      - ``panel.shape[1] == _PANEL_WIDTH``

    左侧总高 = ``_HEADER_HEIGHT + visual_left_bgr.shape[0]``。
    左右两侧高度不一致时,短的一边补背景:左侧补黑、panel 补 panel 的暗灰底色,
    保证最终图无黑边、左右严格对齐。
    """
    target_w = _LEFT_WIDTH

    header = np.zeros((_HEADER_HEIGHT, target_w, 3), dtype=np.uint8)
    _put_text(
        header,
        "ToF 标定可视化(直方图 + 2D 图 + 3D 点云 + 串光/底噪最差像素 bin[0:62])",
        (12, 34), (255, 255, 255), size=22, bold=True,
    )
    left = np.vstack([header, visual_left_bgr])

    left_h = int(left.shape[0])
    panel_h = int(panel.shape[0])
    out_h = max(left_h, panel_h)

    if left_h < out_h:
        pad = np.zeros((out_h - left_h, target_w, 3), dtype=np.uint8)
        left = np.vstack([left, pad])
    if panel_h < out_h:
        # 用 panel 的暗灰底色 (24,24,24) 向下延展,看起来是 panel 自然结束的留白。
        pad = np.full((out_h - panel_h, _PANEL_WIDTH, 3), 24, dtype=np.uint8)
        panel = np.vstack([panel, pad])

    sep = np.full((out_h, _PANEL_SEP_WIDTH, 3), 70, dtype=np.uint8)
    return np.hstack([left, sep, panel])


# ---------------------------------------------------------------------------
# 标定主流程
# ---------------------------------------------------------------------------
def _calibrate(tof_cube: np.ndarray) -> dict[str, Any]:
    depth_map = _depth_from_hist_centroid(tof_cube)
    extra = _compute_extra_metrics(tof_cube)
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

    # 误差 / 偏置统一以 cm 对外暴露；其余产测项见 _compute_extra_metrics。
    values: dict[str, float] = {
        "f":     f,
        "bias":  bias * 100.0,
        "ax":    ax_deg,
        "ay":    ay_deg,
        "rms":   rms_m * 100.0,
        "worst": worst_top_threshold_m * 100.0,
    }
    values.update(extra["values"])

    return {
        "values":         values,
        "residuals":      r_opt,
        "points":         pts_opt,
        "normal":         normal,
        # peak_per_pixel 既是亮度图,也是打光强度的源头。
        "brightness_map":  extra["peak_per_pixel"],
        "bin0_per_pixel":  extra["bin0_per_pixel"],
        "noise_per_pixel": extra["noise_per_pixel"],
        "dead_mask":       extra["dead_mask"],
        "comp_cube":       extra["comp_cube"],
        "crosstalk_top2":  extra["crosstalk_top2"],
        "noise_top2":      extra["noise_top2"],
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
            BGR 图像，左侧为 3x4 标定可视化：
            行 1 = 亮度 / 串光 / 底噪 / 平行度 四张直方图，
            行 2 = 亮度 2D / 串光 2D / 底噪 2D / 3D 点云 + 拟合平面，
            行 3 = 串光最差 #1 / #2、底噪最大 #1 / #2 各像素的 bin[0:62]
                  补偿后柱状图(关注区段橙红高亮);
            右侧为产测项目面板，按"坏点 / 串光 / 底噪 / 打光 / 几何
            标定 / FPPN / 平面度"分组,每一项都列出 measured / threshold /
            状态。
        params : list[float]
            13 个 metric 数值,顺序固定为：
            ``[f(px), ax(deg), ay(deg),
               bias(cm),
               rms(cm), worst(cm),
               dead_pixels,
               crosstalk_max, crosstalk_mean,
               noise_max, noise_mean,
               light_max, light_mean]``。
            其中除几何/平面项之外的 max/mean 都是基于 *最后两个 bin
            饱和补偿后* 的 hist 值。
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

    # 先把右侧 panel 画出来,以它的高度为基准决定左侧 figure 的目标像素,
    # 让左侧从一开始就按"最终像素尺寸"渲染,避免事后 resize 把字体压糊。
    # 3x4 布局每行需要约 300 px 才不挤,所以左侧 body 设了最小高度;若 panel
    # 比左侧短,_compose_combined_image 会在 panel 底部补一段暗灰留白。
    panel = _draw_test_panel(sections, overall_pass, _PANEL_WIDTH)
    target_body_h = max(int(panel.shape[0]) - _HEADER_HEIGHT, _MIN_LEFT_BODY_H)

    visual_left_bgr = _render_visual_left(
        cali["residuals"], cali["points"], cali["normal"],
        cali["brightness_map"], cali["bin0_per_pixel"], cali["noise_per_pixel"],
        cali["comp_cube"], cali["crosstalk_top2"], cali["noise_top2"],
        target_size=(_LEFT_WIDTH, target_body_h),
    )
    image = _compose_combined_image(visual_left_bgr, panel)

    params: list[float] = [float(values[name]) for name in _METRIC_NAMES]
    return bool(overall_pass), image, params

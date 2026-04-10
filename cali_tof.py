#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import json
from pathlib import Path
import cv2
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
import numpy as np

IMG_W = 40
IMG_H = 30
TOF_FRAMES = 64
TOF_HIST_VALID_BINS = 62
TOF_BIN_STEP_M = 0.15 * 4

# 平面到相机原点的固定距离（米）。
PLANE_DISTANCE_M = 1.4

# 固定配置：直接运行脚本即可，不需要命令行参数
DATA_FILE = "tof_60cm.raw"

# ===== 待优化参数初值（宏定义）=====
F_INIT = 47
AX_INIT_DEG = 0.0
AY_INIT_DEG = 0.0

# ===== 待优化参数范围（宏定义）=====
F_MIN = 30
F_MAX = 60.0
AX_MIN_DEG = -15.0
AX_MAX_DEG = 15.0
AY_MIN_DEG = -15.0
AY_MAX_DEG = 15.0

# ===== 优化器参数（宏定义）=====
POWELL_MAXITER = 1000
POWELL_XTOL = 1e-8
POWELL_FTOL = 1e-8
POWELL_DISP = False

# ===== 误差统计参数（宏定义）=====
WORST_ERROR_TOP_RATIO = 0.01

# ===== 亮度统计参数（宏定义）=====
SAT_SCALE = 50000.0
SAT_HIGH_BIN_WEIGHT = 1024.0

# ===== plot缩放尺度 =====
VISUAL_RES_SCALE = 2.0

# ===== 阈值配置文件名（位于本文件同目录）=====
THRESHOLD_JSON_NAME = "threshold.json"
METRIC_NAMES = ("f", "bias", "ax", "ay", "rms", "worst", "peak_mean", "peak_max", "peak_min")

def _build_roi_uv() -> tuple[np.ndarray, np.ndarray]:
    # 生成IMG_HxIMG_W像素网格坐标并展平。
    xs = np.arange(0, IMG_W, dtype=np.float64)
    ys = np.arange(0, IMG_H, dtype=np.float64)
    u, v = np.meshgrid(xs, ys)
    return u.reshape(-1), v.reshape(-1)


def _plane_normal_from_angles(ax_deg: float, ay_deg: float) -> np.ndarray:
    # 由两个“度”单位倾角计算单位平面法向量。
    ax = math.radians(float(ax_deg))
    ay = math.radians(float(ay_deg))
    # 使用两个角度参数控制倾斜量，nz 固定正向，再归一化为单位法向量。
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
    # 把深度图按内参反投影为3D点。
    # 实际距离 = 测量距离 - bias。
    d = depth_flat_m - float(bias)

    uu = u_flat
    vv = v_flat
    dd = d

    x = (uu - float(cx)) / float(f)
    y = (vv - float(cy)) / float(f)

    # 固定使用ray模型：深度视为沿光线距离，需要先归一化方向向量。
    dirs = np.stack([x, y, np.ones_like(x)], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = dirs * dd[:, None]
    return pts


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
    # 计算每个像素点到目标平面的带符号误差。
    f, ax_deg, ay_deg = [float(v) for v in params]
    pts = _points_from_depth(depth_flat_m, u_flat, v_flat, f, cx_fixed, cy_fixed, bias_fixed_m)
    n = _plane_normal_from_angles(ax_deg, ay_deg)
    # 点到平面的有符号距离：n·p - d
    dist = pts @ n - float(plane_distance_m)
    return dist


def _rms(x: np.ndarray) -> float:
    # 计算输入向量的RMS。
    if x.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(x))))


def _load_tof_raw_cube(path: str) -> np.ndarray:
    # 按uint16读取tof.raw，取最后IMG_H*IMG_W*TOF_FRAMES个值并reshape。
    need = IMG_H * IMG_W * TOF_FRAMES
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size < need:
        raise ValueError(f"raw data not enough: need {need}, got {raw.size}")
    cube = raw[-need:].reshape(IMG_H, IMG_W, TOF_FRAMES).astype(np.float32, copy=False)
    return cube


def _depth_from_hist_centroid(tof_cube: np.ndarray) -> np.ndarray:
    # 前62bin找峰值，用峰值左右1bin做三点重心；距离=重心*60cm。
    h = np.asarray(tof_cube, dtype=np.float64)
    if h.shape != (IMG_H, IMG_W, TOF_FRAMES):
        raise ValueError(f"expect cube shape {(IMG_H, IMG_W, TOF_FRAMES)}, got {h.shape}")

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

    # 直接用bin编号做重心，距离=重心bin * 60cm。
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
    # 先做 5x5 均值滤波，再用最近距离减固定平面距离得到 bias。
    filtered = uniform_filter(np.asarray(depth_map_m, dtype=np.float64), size=5, mode="nearest")
    nearest_m = float(np.min(filtered))
    return nearest_m - float(plane_distance_m)


def _compute_peak_brightness(tof_cube: np.ndarray) -> np.ndarray:
    # 参考 view_tof_hist.py：
    # brightness = max(bin0~61) * 50000 / (bin62*1024 + bin63)
    h = np.asarray(tof_cube, dtype=np.float64)
    if h.shape != (IMG_H, IMG_W, TOF_FRAMES):
        raise ValueError(f"expect cube shape {(IMG_H, IMG_W, TOF_FRAMES)}, got {h.shape}")

    peak_first_62 = np.max(h[:, :, :TOF_HIST_VALID_BINS], axis=2)
    denom = h[:, :, 62] * SAT_HIGH_BIN_WEIGHT + h[:, :, 63]
    sat_coeff = np.where(denom > 1e-12, SAT_SCALE / denom, 0.0)
    return peak_first_62 * sat_coeff


def _make_plane_mesh(
    pts: np.ndarray,
    n: np.ndarray,
    d: float,
    scale_pad: float = 0.05,
    min_half_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 根据拟合平面生成用于绘制的网格面。
    # p0 是平面上一点，满足 n·p0 = d。
    p0 = n * float(d)
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(helper, n))) > 0.95:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    e1 = np.cross(n, helper)
    e1 /= max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(n, e1)
    e2 /= max(float(np.linalg.norm(e2)), 1e-12)

    # 根据点云分布自适应设置绘图范围，保证平面覆盖点云区域。
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
    xyz = p0[None, None, :] + aa[..., None] * e1[None, None, :] + bb[..., None] * e2[None, None, :]
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]


def _draw_3d_plot(
    ax: object,
    points: np.ndarray,
    residuals: np.ndarray,
    normal: np.ndarray,
) -> None:
    # 在给定坐标轴上绘制3D点云和平面。
    px, py, pz = _make_plane_mesh(points, normal, PLANE_DISTANCE_M)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=residuals, cmap="coolwarm", s=18, alpha=0.9)
    ax.plot_surface(px, py, pz, alpha=0.35, color="tab:green", linewidth=0, antialiased=True)
    ax.scatter([0.0], [0.0], [0.0], c="k", s=40, marker="x")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("ToF points and calibrated plane")
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _draw_error_distribution_hist(
    ax: object,
    residuals_m: np.ndarray,
) -> None:
    # 在给定坐标轴上绘制误差分布直方图。
    errs_cm = np.asarray(residuals_m, dtype=np.float64).reshape(-1) * 100.0
    errs_cm = errs_cm[np.isfinite(errs_cm)]
    if errs_cm.size == 0:
        return

    ax.hist(errs_cm, bins=30, color="steelblue", edgecolor="white", alpha=0.9)
    ax.set_title("Error distribution")
    ax.set_xlabel("signed error (cm)")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25, linestyle="--")

    mean_cm = float(np.mean(errs_cm))
    std_cm = float(np.std(errs_cm))
    # RMS 用于和终端打印一致对比。
    rms_cm = float(np.sqrt(np.mean(errs_cm * errs_cm)))

    ax.text(
        0.02,
        0.98,
        f"n={errs_cm.size}, rms={rms_cm:.3f} cm",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
    )


def _show_two_plots(residuals_m: np.ndarray, points: np.ndarray, normal: np.ndarray) -> None:
    # 在一个窗口里同时显示误差直方图和3D图。
    fig = plt.figure(figsize=(14, 6))
    ax_hist = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    _draw_error_distribution_hist(ax_hist, residuals_m)
    _draw_3d_plot(ax_3d, points, residuals_m, normal)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def _fig_to_rgb_image(fig: object) -> np.ndarray:
    # 把 matplotlib Figure 渲染为 HxWx3 的 uint8 RGB 图像。
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return np.asarray(buf[:, :, :3], dtype=np.uint8)


def _render_hist_image(residuals_m: np.ndarray) -> np.ndarray:
    # 渲染误差直方图并返回图像。
    fig = plt.figure(figsize=(7 * VISUAL_RES_SCALE, 6 * VISUAL_RES_SCALE))
    ax = fig.add_subplot(1, 1, 1)
    _draw_error_distribution_hist(ax, residuals_m)
    fig.tight_layout()
    img = _fig_to_rgb_image(fig)
    plt.close(fig)
    return img


def _render_3d_image(points: np.ndarray, residuals_m: np.ndarray, normal: np.ndarray) -> np.ndarray:
    # 渲染3D点云+平面图并返回图像。
    fig = plt.figure(figsize=(7 * VISUAL_RES_SCALE, 6 * VISUAL_RES_SCALE))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _draw_3d_plot(ax, points, residuals_m, normal)
    fig.tight_layout()
    img = _fig_to_rgb_image(fig)
    plt.close(fig)
    return img

def calibrate_tof(
    data_file: str = DATA_FILE,
    interactive_show: bool = False,
) -> dict[str, object]:
    # 执行标定并返回关键结果。
    tof_cube = _load_tof_raw_cube(data_file)
    # 深度计算：前62bin峰值 + 左中右三点重心，再乘60cm。
    depth_map = _depth_from_hist_centroid(tof_cube)
    peak_brightness = _compute_peak_brightness(tof_cube)
    bias_fixed = _compute_bias_from_depth(depth_map, PLANE_DISTANCE_M)
    depth_flat = depth_map.reshape(-1)
    u_flat, v_flat = _build_roi_uv()

    # 光心固定在图像中心。
    cx0 = (IMG_W - 1) / 2.0
    cy0 = (IMG_H - 1) / 2.0

    # 参数顺序: [f, ax, ay]，cx/cy 固定在图像中心，bias 由数据计算固定。
    x0 = np.array([F_INIT, AX_INIT_DEG, AY_INIT_DEG], dtype=np.float64)
    bounds = [(F_MIN, F_MAX), (AX_MIN_DEG, AX_MAX_DEG), (AY_MIN_DEG, AY_MAX_DEG)]

    def objective_rms(p: np.ndarray) -> float:
        # 优化目标：最小化所有像素点到平面的RMS误差。
        r = _residuals(p, depth_flat, u_flat, v_flat, cx0, cy0, PLANE_DISTANCE_M, bias_fixed)
        return _rms(r)

    # Powell：不需要梯度，适合这个小维度问题快速试参。
    pw_res = minimize(
        objective_rms,
        x0=np.asarray(x0, dtype=np.float64),
        method="Powell",
        bounds=bounds,
        options={"maxiter": POWELL_MAXITER, "xtol": POWELL_XTOL, "ftol": POWELL_FTOL, "disp": POWELL_DISP},
    )

    # 取出优化后的最优参数，并重新计算全量残差与RMS。
    x_opt = np.asarray(pw_res.x, dtype=np.float64)
    r_opt = _residuals(x_opt, depth_flat, u_flat, v_flat, cx0, cy0, PLANE_DISTANCE_M, bias_fixed)
    rms_m = _rms(r_opt)

    # 解包参数并转换角度单位，便于打印阅读。
    f, ax_deg, ay_deg = [float(v) for v in x_opt]
    bias = float(bias_fixed)
    cx = float(cx0)
    cy = float(cy0)
    normal = _plane_normal_from_angles(ax_deg, ay_deg)

    # 用最优参数重建点云并统计结果。
    pts_opt = _points_from_depth(depth_flat, u_flat, v_flat, f, cx, cy, bias)
    residual_valid = r_opt
    abs_err = np.abs(residual_valid)
    worst_k = max(1, int(math.ceil(abs_err.size * WORST_ERROR_TOP_RATIO)))
    # 绝对误差降序后，第worst_k个值作为“最坏1%”的误差阈值。
    worst_top_threshold_m = float(np.partition(abs_err, abs_err.size - worst_k)[abs_err.size - worst_k])
    pb_flat = np.asarray(peak_brightness, dtype=np.float64).reshape(-1)
    pb_flat = pb_flat[np.isfinite(pb_flat)]
    pb_mean = float(np.mean(pb_flat)) if pb_flat.size > 0 else 0.0
    pb_max = float(np.max(pb_flat)) if pb_flat.size > 0 else 0.0
    pb_min = float(np.min(pb_flat)) if pb_flat.size > 0 else 0.0

    visual_result = np.concatenate(
        [
            _render_hist_image(residual_valid),
            _render_3d_image(pts_opt, residual_valid, normal),
        ],
        axis=1,
    )

    result = {
        "f": f,
        "bias": bias,
        "ax": ax_deg,
        "ay": ay_deg,
        "rms": rms_m,
        "worst": worst_top_threshold_m,
        "peak_mean": pb_mean,
        "peak_max": pb_max,
        "peak_min": pb_min,
        "visual_result": visual_result,
    }

    if interactive_show:
        # 可选交互显示，不影响返回的visual_result。
        _show_two_plots(residual_valid, pts_opt, normal)

    return result


def _metric_in_range(value: float, cfg: dict[str, float]) -> tuple[bool, str]:
    # 只支持且要求同时提供 min / max。
    if "min" not in cfg or "max" not in cfg:
        raise ValueError("threshold config must contain both 'min' and 'max'")
    min_v = float(cfg["min"])
    max_v = float(cfg["max"])
    if value < min_v:
        return False, f"{value:.4f}, range=[{min_v:.4f}, {max_v:.4f}]"
    if value > max_v:
        return False, f"{value:.4f}, range=[{min_v:.4f}, {max_v:.4f}]"
    return True, ""


def check(raw_file: str) -> tuple[bool, str, list[float], np.ndarray]:
    # 调用标定并根据阈值文件做合格性检查，然后打印固定格式结果。
    cali_res = calibrate_tof(raw_file, interactive_show=False)

    threshold_path = Path(__file__).resolve().parent / THRESHOLD_JSON_NAME
    with open(threshold_path, "r", encoding="utf-8") as fp:
        threshold_cfg = json.load(fp)

    issues: list[str] = []
    failed_metric_names: set[str] = set()
    for name in METRIC_NAMES:
        if name not in threshold_cfg:
            raise ValueError(f"missing threshold for metric '{name}'")
        ok, reason = _metric_in_range(float(cali_res[name]), threshold_cfg[name])
        if not ok:
            issues.append(f"{name} = {reason}")
            failed_metric_names.add(name)

    is_pass = len(issues) == 0
    desc = "OK" if is_pass else "; ".join(issues)
    data = [float(cali_res[name]) for name in METRIC_NAMES]
    visual_result = np.asarray(cali_res["visual_result"])

    print(f"pass      : {is_pass}")
    print(f"desc      : {desc}")
    lines = [f"pass      : {is_pass}", f"desc      : {desc}"]
    for name, value in zip(METRIC_NAMES, data):
        min_v = float(threshold_cfg[name]["min"])
        max_v = float(threshold_cfg[name]["max"])
        line = f"{name:10s}: {value:10.4f}   thr:[{min_v:8.4f}, {max_v:8.4f}]"
        lines.append(line)
        print(line)

    # 在图像上方新增黑色信息栏，避免遮挡原图内容。
    bgr = cv2.cvtColor(visual_result, cv2.COLOR_RGB2BGR)
    info_h = 20 + 30 * len(lines)
    canvas = np.zeros((bgr.shape[0] + info_h, bgr.shape[1], 3), dtype=np.uint8)
    canvas[info_h:, :, :] = bgr
    title_color = (0, 220, 0) if is_pass else (0, 0, 255)
    for idx, line in enumerate(lines):
        y = 30 + idx * 30
        if idx == 0:
            color = title_color
        elif idx == 1:
            color = (255, 255, 255)
        else:
            metric_name = METRIC_NAMES[idx - 2]
            color = (0, 0, 255) if metric_name in failed_metric_names else (255, 255, 255)
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    visual_result = canvas

    return is_pass, desc, data, visual_result

if __name__ == "__main__":
    calibrate_tof(data_file=DATA_FILE)


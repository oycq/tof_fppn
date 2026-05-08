#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实时读取 ToF 数据，仅调用 tof_fppn 包的 run_all_checks 接口：
- 几何标定 / FPPN / 平面度 / 坏点 / 串光 / 底噪 / 打光强度

单线程主循环：trigger -> pull -> check -> 显示。
所有外部子进程超时统一为 0.3 秒，任何一步超时/失败都丢这帧继续下一帧。
"""

from __future__ import annotations

import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from tof_fppn import run_all_checks


TOF_H = 30
TOF_W = 40
TOF_C = 64
TOF_RAW_HEADER_BYTES = 5120
RAW_EXPECTED_BYTES = TOF_RAW_HEADER_BYTES + TOF_H * TOF_W * TOF_C * 2

STEP_TIMEOUT_S = 0.3

TMP_DIR = Path("./tmp")
TMP_PULL_RAW_PATH = TMP_DIR / "tof_pull.raw"
TMP_CHECK_RAW_PATH = TMP_DIR / "realtime_check.raw"
SNAPSHOTS_DIR = Path("./snapshots")

# tof_fppn 输出图宽度固定为 1800；高度随面板/可视化变化。
PLACEHOLDER_H = 600
PLACEHOLDER_W = 1800


def _placeholder_view(h: int = PLACEHOLDER_H, w: int = PLACEHOLDER_W) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _overlay_saved(view: np.ndarray, text: str = "SAVED") -> np.ndarray:
    out = view.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.6
    thickness = 4
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cx, cy = w // 2, h // 2
    pad = 20
    x1, y1 = cx - tw // 2 - pad, cy - th // 2 - pad
    x2, y2 = cx + tw // 2 + pad, cy + th // 2 + pad
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, dst=out)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        text,
        (cx - tw // 2, cy + th // 2),
        font,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )
    return out


def _save_snapshot(
    snapshots_dir: Path,
    image: np.ndarray | None,
    raw_bytes: bytes | None,
) -> Path | None:
    if image is None and raw_bytes is None:
        return None
    folder = snapshots_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    # 极少数情况下同一秒按两次，避免覆盖
    if folder.exists():
        folder = snapshots_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=True)

    if image is not None:
        cv2.imwrite(str(folder / "view.png"), image)

    if raw_bytes is not None:
        raw_path = folder / "tof.raw"
        raw_path.write_bytes(raw_bytes)

    return folder


def _check_adb_connected() -> bool:
    try:
        r = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            timeout=3.0,
            check=False,
            text=True,
        )
    except Exception:
        return False
    if r.returncode != 0:
        return False
    for ln in (r.stdout or "").splitlines()[1:]:
        if "\tdevice" in ln:
            return True
    return False


def _adb_trigger() -> tuple[bool, str]:
    cmd = "if [ -e /tmp/sv_tof ]; then rm /tmp/sv_tof && rm /tmp/tof.raw; fi && touch /tmp/sv_tof"
    try:
        r = subprocess.run(
            ["adb", "shell", cmd],
            timeout=STEP_TIMEOUT_S,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if int(r.returncode) == 0:
            return True, ""
        err = (r.stderr or b"").decode("utf-8", errors="ignore").strip()
        return False, f"adb shell rc={r.returncode}: {err[:200]}"
    except subprocess.TimeoutExpired:
        return False, "adb shell timeout"
    except Exception as e:
        return False, f"adb shell exc: {e!r}"


def _adb_pull_raw() -> tuple[bytes | None, str]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if TMP_PULL_RAW_PATH.exists():
            TMP_PULL_RAW_PATH.unlink(missing_ok=True)
        r = subprocess.run(
            ["adb", "pull", "/tmp/tof.raw", str(TMP_PULL_RAW_PATH)],
            timeout=STEP_TIMEOUT_S,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if int(r.returncode) != 0:
            err = (r.stderr or b"").decode("utf-8", errors="ignore").strip() or f"rc={r.returncode}"
            return None, err
        if not TMP_PULL_RAW_PATH.exists():
            return None, "pulled file missing"
        size = int(TMP_PULL_RAW_PATH.stat().st_size)
        if size < RAW_EXPECTED_BYTES:
            return None, f"pulled too small: {size} < {RAW_EXPECTED_BYTES}"
        out = TMP_PULL_RAW_PATH.read_bytes()
        return bytes(out[:RAW_EXPECTED_BYTES]), ""
    except subprocess.TimeoutExpired:
        return None, "adb pull timeout"
    except Exception as e:
        return None, f"adb pull exc: {e!r}"


def _throttle_log(state: dict, msg: str) -> None:
    now = time.time()
    if now - state.get("last_log_ts", 0.0) > 2.0:
        print(f"[realtime] {msg}", flush=True)
        state["last_log_ts"] = now


def main() -> int:
    if not _check_adb_connected():
        print("[realtime] adb 无法连接（请检查设备是否插好、是否授权）", flush=True)
        return 1

    window_name = "TOF_FPPN_REALTIME_CHECK"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    image_cache: np.ndarray | None = None
    latest_raw_bytes: bytes | None = None
    view_cache: np.ndarray = _placeholder_view()
    log_state: dict = {"last_log_ts": 0.0}
    saved_flash_until: float = 0.0
    saved_flash_text: str = ""

    try:
        while True:
            ok, err = _adb_trigger()
            if ok:
                raw_bytes, perr = _adb_pull_raw()
                if raw_bytes is not None:
                    latest_raw_bytes = raw_bytes
                    try:
                        TMP_CHECK_RAW_PATH.write_bytes(raw_bytes)
                        _passed, image, _params = run_all_checks(str(TMP_CHECK_RAW_PATH))
                        if image is not None:
                            image_cache = image
                    except Exception:
                        traceback.print_exc()
                else:
                    _throttle_log(log_state, f"pull 失败: {perr}")
            else:
                _throttle_log(log_state, f"trigger 失败: {err}")

            base_view = image_cache if image_cache is not None else _placeholder_view()
            if time.time() < saved_flash_until:
                view_cache = _overlay_saved(base_view, saved_flash_text or "SAVED")
            else:
                view_cache = base_view
            cv2.imshow(window_name, view_cache)

            key = int(cv2.waitKey(1) & 0xFF)
            if key == 32:  # Space: 保存当前快照（view + raw）到 snapshots/<时间>/
                try:
                    folder = _save_snapshot(SNAPSHOTS_DIR, image_cache, latest_raw_bytes)
                    if folder is not None:
                        print(f"[存储成功] 快照已保存: {folder}")
                        saved_flash_text = "SAVED"
                        saved_flash_until = time.time() + 1.0
                    else:
                        print("[realtime] 暂无可保存的数据")
                        saved_flash_text = "NO DATA"
                        saved_flash_until = time.time() + 1.0
                except Exception:
                    traceback.print_exc()
                    saved_flash_text = "SAVE FAILED"
                    saved_flash_until = time.time() + 1.0
            if key == 48:  # '0': save current raw
                try:
                    if latest_raw_bytes is not None:
                        TMP_DIR.mkdir(parents=True, exist_ok=True)
                        raw_path = TMP_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.raw"
                        raw_path.write_bytes(latest_raw_bytes)
                        print(f"[存储成功] RAW已保存: {raw_path}")
                except Exception:
                    pass
            if key == 27:  # ESC
                break
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

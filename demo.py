#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import cv2
import numpy as np

from cali_tof import calibrate_tof


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ToF calibration demo and show visual_result.")
    parser.add_argument("--data-file", default="tof_60cm.raw", help="Path to raw ToF file.")
    parser.add_argument("--no-show", action="store_true", help="Do not open image window.")
    args = parser.parse_args()

    result = calibrate_tof(args.data_file)
    visual_result = np.asarray(result["visual_result"])

    print("\n=== demo.py return fields ===")
    for k in ("f", "bias", "ax", "ay", "rms", "worst", "peak_mean", "peak_max", "peak_min"):
        print(f"{k:10s}: {result[k]}")
    print(f"visual_result shape: {visual_result.shape}, dtype: {visual_result.dtype}")

    if args.no_show:
        return

    bgr = cv2.cvtColor(visual_result, cv2.COLOR_RGB2BGR)
    cv2.imshow("calibrate_tof visual_result", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

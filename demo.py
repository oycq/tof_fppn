#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from cali_tof import check

# ===== 宏定义配置 =====
DATA_FILE = "tof_60cm.raw"


if __name__ == "__main__":
    is_pass, desc, data, visual_result = check(DATA_FILE)
    visual_result = np.asarray(visual_result)
    print(f"\ndemo.py output:")
    print(f"pass      : {is_pass}")
    print(f"desc      : {desc}")
    print(f"data      : {[round(v, 4) for v in data]}")

    cv2.imshow("calibrate_tof visual_result", visual_result)
    cv2.waitKey(0)

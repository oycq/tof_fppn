"""命令行入口：调用 tof_fppn 包，显示拼接结果图。

用法::

    python run.py tof_60cm.raw
    python run.py tof_15cm.raw
"""

import sys

import cv2

from tof_fppn import run_all_checks


def main() -> int:
    raw_path = sys.argv[1]
    passed, image, params = run_all_checks(raw_path)

    print(f"pass   : {passed}")
    print(f"params : {[round(v, 4) for v in params]}")

    cv2.imshow("tof_fppn", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
tof_fppn
========

ToF FPPN（Fixed-Pattern Plane Noise）/ 几何标定包。
一次调用同时完成 **几何标定**、**平面拟合误差**、**光度统计** 三类产测，
返回总判定、可视化结果图，以及一组结构化数值。

用法::

    from tof_fppn import run_all_checks

    # tof.raw 路径相对"调用时 Python 的当前工作目录"
    # 中间产物会落到 tof_fppn/tmp/ 内，不污染调用方目录
    passed, image, params = run_all_checks("tof_60cm.raw")
    # passed : bool
    # image  : numpy.ndarray (H, W, 3) BGR，可直接 cv2.imshow / cv2.imwrite
    # params : list[float] 长度 9
    #          [f(px), bias(cm), ax(deg), ay(deg),
    #           rms(cm), worst(cm), peak_mean, peak_max, peak_min]

返回结构详见 :func:`run_all_checks`。
"""

from ._runner import run_all_checks

__all__ = ["run_all_checks"]

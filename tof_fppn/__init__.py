"""
tof_fppn
========

ToF FPPN（Fixed-Pattern Plane Noise）/ 几何标定 / 光度产测包。
一次调用完成 7 类产测：

  - 几何标定（焦距 / 倾角）
  - FPPN 检测（距离偏置）
  - 平面度（残差均值 / 最坏 1%）
  - 坏点检测（前 62 bin 全 0 像素数）
  - 串光检测（bin[0] 补偿值的 max/mean）
  - 底噪检测（bin[30:50] 补偿值的 max/mean）
  - 打光强度（每像素峰值 bin 补偿值的 max/mean/min）

用法::

    from tof_fppn import run_all_checks

    passed, image, params = run_all_checks("tof.raw")
    # passed : bool
    # image  : numpy.ndarray (H, W, 3) BGR，可直接 cv2.imshow / cv2.imwrite
    # params : list[float] 长度 14
    #          [f(px), ax(deg), ay(deg),
    #           bias(cm),
    #           rms(cm), worst(cm),
    #           dead_pixels,
    #           crosstalk_max, crosstalk_mean,
    #           noise_max, noise_mean,
    #           light_max, light_mean, light_min]

返回结构详见 :func:`run_all_checks`。
"""

from ._runner import run_all_checks

__all__ = ["run_all_checks"]

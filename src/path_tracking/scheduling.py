"""Stanley 增益速度调度（v1.2）：纯函数。"""
from __future__ import annotations

import numpy as np


def stanley_gain_schedule(
    v: float,
    k0: float,
    mode: str = "const",
    v_ref: float = 5.5556,
    alpha: float = 0.05,
    k_soft: float = 1.0,
) -> float:
    """按当前速度返回 Stanley 有效增益 k_eff。

    mode:
      const      k_eff = k0（PythonRobotics 原版）
      linear     k_eff = k0·(1 + alpha·(v − v_ref))  高速增强反馈
      speed_inv  k_eff = k0·(v + k_soft)/(v_ref + k_soft)
                 使反馈项 k_eff·e/(v+k_soft) = k0·e/(v_ref+k_soft) 速度不变
    """
    v = float(np.clip(v, 0.0, None))
    if mode == "const":
        return float(k0)
    if mode == "linear":
        return float(k0 * (1.0 + alpha * (v - v_ref)))
    if mode == "speed_inv":
        return float(k0 * (v + k_soft) / (v_ref + k_soft))
    raise ValueError(f"未知调度模式: {mode}")

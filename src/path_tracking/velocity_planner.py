"""轨迹级速度规划（v1.3，回应报告 §4.5 "轨迹级速度规划"）。

从 v1.2 E9 失效包络（108 格）反推安全速度剖面：
  v_safe(R, σ) = min(v_max, sqrt(μ_eff · g · R) · decay(σ))
其中 decay(σ) 是噪声衰减因子——σ 越大，允许速度越低。

设计依据（E9 数据）：
  - 无噪声：R=15 m 时 ≤20 km/h 安全；R=30/60 m 全速度安全
  - σ=5 cm：仅 R≥30 m + ≤20 km/h 存活
  - σ≥10 cm：全灭 → v_safe ≈ 0（或极低速）
"""

from __future__ import annotations

import numpy as np


def noise_decay(sigma_m: float) -> float:
    """噪声衰减因子 ∈ (0, 1]。σ=0 → 1.0；σ 越大越小。

    拟合 E9 包络的分段指数衰减：
      decay(σ) = exp(−(σ/σ_ref)²)  with σ_ref = 0.04 m
    σ=0 → 1.0, σ=0.05 → 0.21, σ=0.10 → 0.002
    """
    sigma_ref = 0.04
    return float(np.exp(-(float(sigma_m) / sigma_ref) ** 2))


def safe_speed(radius: float, sigma_m: float = 0.0,
               mu_eff: float = 0.35, g: float = 9.81,
               v_max: float = 16.667, v_min: float = 1.0) -> float:
    """给定曲率半径与噪声水平，返回安全速度 [m/s]。

    mu_eff: 有效摩擦系数（含侧向安全裕度，比物理 μ 小）。
    v_min: 最低巡航速度（避免完全停车）。
    """
    if radius <= 0:
        raise ValueError("radius 必须 > 0")
    v_curve = np.sqrt(mu_eff * g * radius)
    v_noise = v_curve * noise_decay(sigma_m)
    return float(np.clip(v_noise, v_min, v_max))


def plan_speed_profile(curvature: np.ndarray, sigma_m: float = 0.0,
                       ds: float = 0.5, a_max: float = 2.0,
                       **kwargs) -> np.ndarray:
    """沿路径生成速度剖面（含加减速约束平滑）。

    curvature: 逐点曲率 κ（1/m），radius = 1/|κ|（κ=0 → radius=inf → v_max）。
    ds: 采样间距 [m]。
    a_max: 最大加/减速度 [m/s²]（平滑限幅用）。
    返回逐点目标速度 [m/s]，长度 = len(curvature)。
    """
    kappa = np.asarray(curvature, dtype=float).reshape(-1)
    n = kappa.size
    v_raw = np.empty(n)
    for i in range(n):
        k = abs(kappa[i])
        R = 1e9 if k < 1e-6 else 1.0 / k
        v_raw[i] = safe_speed(R, sigma_m, **kwargs)

    # 后向减速约束：v[i] ≤ sqrt(v[i+1]² + 2·a_max·ds)
    v = v_raw.copy()
    for i in range(n - 2, -1, -1):
        v_limit = np.sqrt(v[i + 1] ** 2 + 2.0 * a_max * ds)
        v[i] = min(v[i], v_limit)
    # 前向加速约束：v[i+1] ≤ sqrt(v[i]² + 2·a_max·ds)
    for i in range(1, n):
        v_limit = np.sqrt(v[i - 1] ** 2 + 2.0 * a_max * ds)
        v[i] = min(v[i], v_limit)
    return v

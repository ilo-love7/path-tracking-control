"""观测滤波与最近点滞回（v1.3，回应报告 §4.5 "观测滤波/滞回（E5/E9 噪声崩坏的工程解）"）。

两个正交组件：
  1. StateFilter —— EMA 低通滤波器，对含噪观测 (x, y, yaw, v) 平滑；
  2. HysteresisNearest —— 带滞回的最近点索引，抑制索引跳动（PP 噪声死穴的直接机制）。
"""

from __future__ import annotations

import numpy as np


class StateFilter:
    """指数移动平均（EMA）低通滤波器，逐分量独立平滑。

    一阶 IIR：x̂_t = α · z_t + (1 − α) · x̂_{t−1}
    等效截止频率 f_c = −ln(1−α) / (2π·dt)；α 越小越平滑、相位滞后越大。
    yaw 分量做角度环绕处理（差值 wrap 到 [−π, π]）。
    """

    def __init__(self, alpha: float = 0.3, n_channels: int = 4):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha 必须在 (0, 1]，收到 {alpha}")
        self.alpha = float(alpha)
        self.n_channels = int(n_channels)
        self._state = np.zeros(self.n_channels)
        self._initialized = False

    def reset(self):
        self._state[:] = 0.0
        self._initialized = False

    def update(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.size != self.n_channels:
            raise ValueError(f"观测维度 {z.size} ≠ n_channels {self.n_channels}")
        if not self._initialized:
            self._state = z.copy()
            self._initialized = True
            return self._state.copy()
        # yaw（index 2）做角度环绕差
        delta = z - self._state
        delta[2] = (delta[2] + np.pi) % (2 * np.pi) - np.pi
        self._state = self._state + self.alpha * delta
        self._state[2] = (self._state[2] + np.pi) % (2 * np.pi) - np.pi
        return self._state.copy()

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()


class HysteresisNearest:
    """带滞回的最近点索引跟踪器。

    机制：
      - 在 [cur − window, cur + window] 内找最近点（而非全路径）；
      - 若新索引 < cur（回退），只有当回退距离超过 hysteresis 才接受，
        否则维持 cur——抑制噪声引起的索引抖动/跳动。
    """

    def __init__(self, window: int = 30, hysteresis: int = 3):
        if window < 1:
            raise ValueError("window 必须 ≥ 1")
        if hysteresis < 0:
            raise ValueError("hysteresis 必须 ≥ 0")
        self.window = int(window)
        self.hysteresis = int(hysteresis)
        self._idx = 0

    def reset(self, idx: int = 0):
        self._idx = int(idx)

    @property
    def index(self) -> int:
        return self._idx

    def update(self, path, x: float, y: float) -> int:
        """在滞回窗口内更新索引。path 需有 .x/.y 数组。"""
        n = len(path.x)
        lo = max(0, self._idx - self.window)
        hi = min(n, self._idx + self.window + 1)
        dx = path.x[lo:hi] - x
        dy = path.y[lo:hi] - y
        local_best = lo + int(np.argmin(dx * dx + dy * dy))
        if local_best < self._idx - self.hysteresis:
            self._idx = local_best  # 大幅回退才接受
        elif local_best >= self._idx:
            self._idx = local_best  # 前进总是接受
        # 小幅回退（−hysteresis < local_best − cur < 0）→ 维持 cur
        return self._idx

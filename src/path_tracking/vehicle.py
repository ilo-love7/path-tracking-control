"""车辆模型：运动学自行车模型（Kong et al., IEEE IV 2015）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0


class KinematicBicycle:
    """后轮中心运动学自行车模型。

    状态 ξ = [x, y, yaw, v]，输入 u = [accel, steer]。
        ẋ = v cos(yaw)
        ẏ = v sin(yaw)
        ψ̇ = v tan(δ) / L
        v̇ = a
    """

    def __init__(self, wheelbase: float = 2.9, max_steer: float = 0.7854,
                 max_steer_rate: float = 0.5236, dt: float = 0.1):
        self.L = float(wheelbase)
        self.max_steer = float(max_steer)
        self.max_steer_rate = float(max_steer_rate)
        self.dt = float(dt)
        self.state = VehicleState()
        self.prev_steer = 0.0

    def reset(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.state = VehicleState(float(x), float(y), float(yaw), float(v))
        self.prev_steer = 0.0
        return self.state

    def _clip_steer(self, steer: float, dt: float | None = None) -> float:
        dt = self.dt if dt is None else dt
        # 转角速率限制
        max_d = self.max_steer_rate * dt
        steer = float(np.clip(steer, self.prev_steer - max_d, self.prev_steer + max_d))
        # 转角幅值限制
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        return steer

    def step(self, accel: float, steer: float, dt: float | None = None) -> VehicleState:
        """RK4 积分一步。返回新状态。"""
        dt = self.dt if dt is None else dt
        steer = self._clip_steer(steer, dt)
        self.prev_steer = steer

        def f(s: VehicleState):
            return np.array([
                s.v * np.cos(s.yaw),
                s.v * np.sin(s.yaw),
                s.v * np.tan(steer) / self.L,
                accel,
            ])

        s0 = np.array([self.state.x, self.state.y, self.state.yaw, self.state.v])
        k1 = f(self.state)
        s2 = VehicleState(*(s0 + 0.5 * dt * k1))
        k2 = f(s2)
        s3 = VehicleState(*(s0 + 0.5 * dt * k2))
        k3 = f(s3)
        s4 = VehicleState(*(s0 + dt * k3))
        k4 = f(s4)
        s_next = s0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        self.state = VehicleState(*[float(x) for x in s_next])
        return self.state

    @property
    def state_vector(self) -> np.ndarray:
        return np.array([self.state.x, self.state.y, self.state.v, self.state.yaw])

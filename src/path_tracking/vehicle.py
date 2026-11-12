"""车辆模型：运动学自行车（Kong et al., IEEE IV 2015）+ 动力学自行车（v1.1）。

动力学模型（线性轮胎 2-DOF，Kong 2015 dynamic bicycle）：
  状态 [x, y, v, yaw, vy, r]，输入 [a, δ]
    ẋ  = v cos(yaw) - vy sin(yaw)
    ẏ  = v sin(yaw) + vy cos(yaw)
    v̇  = a
    ψ̇  = r
    v̇y = (Fyf + Fyr)/m - v·r
    ṙ  = (lf·Fyf - lr·Fyr)/Iz
  轮胎力（线性侧偏刚度）：
    Fyf = Cf·(δ - atan((vy + lf·r)/max(v, 0.5)))
    Fyr = Cr·(-atan((vy - lr·r)/max(v, 0.5)))
"""

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


class DynamicBicycle:
    """2-DOF 动力学自行车模型（线性轮胎，Kong 2015 dynamic）。

    状态 ξ = [x, y, v, yaw, vy, r]，输入 u = [accel, steer]。
    与 KinematicBicycle 共享限幅与 reset 语义；state 属性暴露 [x,y,yaw,v]，
    额外侧偏 vy / 横摆角速度 r 存在 .beta/.yaw_rate。
    """

    def __init__(self, wheelbase: float = 2.9, max_steer: float = 0.7854,
                 max_steer_rate: float = 0.5236, dt: float = 0.1,
                 mass: float = 1573.0, iz: float = 2500.0,
                 lf: float = 1.2, lr: float = 1.7,
                 cf: float = 80000.0, cr: float = 80000.0,
                 wind_accel: float = 0.0):
        self.L = float(wheelbase)
        self.max_steer = float(max_steer)
        self.max_steer_rate = float(max_steer_rate)
        self.dt = float(dt)
        self.m = float(mass)
        self.Iz = float(iz)
        self.lf = float(lf)
        self.lr = float(lr)
        self.Cf = float(cf)
        self.Cr = float(cr)
        self.wind_accel = float(wind_accel)  # 侧风等效横向加速度 [m/s^2]
        self.state = VehicleState()
        self.vy = 0.0
        self.r = 0.0
        self.prev_steer = 0.0

    def reset(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.state = VehicleState(float(x), float(y), float(yaw), float(v))
        self.vy = 0.0
        self.r = 0.0
        self.prev_steer = 0.0
        return self.state

    def _clip_steer(self, steer: float, dt: float | None = None) -> float:
        dt = self.dt if dt is None else dt
        max_d = self.max_steer_rate * dt
        steer = float(np.clip(steer, self.prev_steer - max_d, self.prev_steer + max_d))
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        return steer

    def step(self, accel: float, steer: float, dt: float | None = None) -> VehicleState:
        """RK4 积分一步。返回新状态。"""
        dt = self.dt if dt is None else dt
        steer = self._clip_steer(steer, dt)
        self.prev_steer = steer

        def f(s: np.ndarray) -> np.ndarray:
            x, y, v, yaw, vy, r = s
            vx = max(v, 0.5)  # 低速保护，避免侧偏角发散
            alpha_f = steer - np.arctan2(vy + self.lf * r, vx)
            alpha_r = -np.arctan2(vy - self.lr * r, vx)
            fyf = self.Cf * alpha_f
            fyr = self.Cr * alpha_r
            return np.array([
                v * np.cos(yaw) - vy * np.sin(yaw),
                v * np.sin(yaw) + vy * np.cos(yaw),
                accel,
                r,
                (fyf + fyr) / self.m - v * r + self.wind_accel,
                (self.lf * fyf - self.lr * fyr) / self.Iz,
            ])

        s0 = np.array([self.state.x, self.state.y, self.state.v, self.state.yaw,
                       self.vy, self.r])
        k1 = f(s0)
        k2 = f(s0 + 0.5 * dt * k1)
        k3 = f(s0 + 0.5 * dt * k2)
        k4 = f(s0 + dt * k3)
        s_next = s0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        self.state = VehicleState(float(s_next[0]), float(s_next[1]),
                                  float(s_next[3]), float(s_next[2]))
        self.vy = float(s_next[4])
        self.r = float(s_next[5])
        return self.state

    @property
    def state_vector(self) -> np.ndarray:
        return np.array([self.state.x, self.state.y, self.state.v, self.state.yaw,
                         self.vy, self.r])


class NonlinearDynamicBicycle(DynamicBicycle):
    """Pacejka-lite 饱和轮胎动力学自行车（v1.3）。

    与 DynamicBicycle 的唯一区别：侧偏力从线性 Fy = C·α 改为饱和曲线
        Fy = Fy_max · tanh(C·α / Fy_max)
    其中 Fy_max = μ·Fz（静态轴荷分配 Fz_f = m·g·lr/L，Fz_r = m·g·lf/L）。
    小侧偏角（|C·α| << Fy_max）退化为线性；大侧偏角饱和在 μ·Fz。
    """

    def __init__(self, *args, mu: float = 0.9, g: float = 9.81, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = float(mu)
        self.g = float(g)
        L = self.lf + self.lr
        self._Fz_f = self.m * self.g * self.lr / L
        self._Fz_r = self.m * self.g * self.lf / L
        self._Fyf_max = self.mu * self._Fz_f
        self._Fyr_max = self.mu * self._Fz_r

    def step(self, accel: float, steer: float, dt: float | None = None) -> VehicleState:
        dt = self.dt if dt is None else dt
        steer = self._clip_steer(steer, dt)
        self.prev_steer = steer

        def f(s: np.ndarray) -> np.ndarray:
            x, y, v, yaw, vy, r = s
            vx = max(v, 0.5)
            alpha_f = steer - np.arctan2(vy + self.lf * r, vx)
            alpha_r = -np.arctan2(vy - self.lr * r, vx)
            # Pacejka-lite 饱和
            fyf = self._Fyf_max * np.tanh(self.Cf * alpha_f / self._Fyf_max)
            fyr = self._Fyr_max * np.tanh(self.Cr * alpha_r / self._Fyr_max)
            return np.array([
                v * np.cos(yaw) - vy * np.sin(yaw),
                v * np.sin(yaw) + vy * np.cos(yaw),
                accel,
                r,
                (fyf + fyr) / self.m - v * r + self.wind_accel,
                (self.lf * fyf - self.lr * fyr) / self.Iz,
            ])

        s0 = np.array([self.state.x, self.state.y, self.state.v, self.state.yaw,
                       self.vy, self.r])
        k1 = f(s0)
        k2 = f(s0 + 0.5 * dt * k1)
        k3 = f(s0 + 0.5 * dt * k2)
        k4 = f(s0 + dt * k3)
        s_next = s0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        self.state = VehicleState(float(s_next[0]), float(s_next[1]),
                                  float(s_next[3]), float(s_next[2]))
        self.vy = float(s_next[4])
        self.r = float(s_next[5])
        return self.state

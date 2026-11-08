"""参考路径生成：双移线 / 环形 / 三次样条平滑。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Path:
    x: np.ndarray        # (N,)
    y: np.ndarray        # (N,)
    yaw: np.ndarray      # (N,)
    curvature: np.ndarray
    s: np.ndarray        # 弧长 [m]

    def __len__(self):
        return len(self.x)


def _arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ds = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(ds)])


def _yaw_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx, dy = np.diff(x), np.diff(y)
    yaw = np.arctan2(dy, dx)
    return np.concatenate([yaw, [yaw[-1]]])


def _curvature_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num = dx * ddy - dy * ddx
    den = np.power(dx * dx + dy * dy, 1.5)
    den[den == 0] = 1e-9
    return num / den


class CubicSpline2D:
    """参数化弧长三次样条（对齐 PythonRobotics cubic_spline_planner）。"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        s = _arc_length(x, y)
        self.s = s
        self.sx = _CubicSpline(s, x)
        self.sy = _CubicSpline(s, y)

    def position(self, s_q: float | np.ndarray):
        return self.sx(s_q), self.sy(s_q)

    def yaw(self, s_q: float | np.ndarray):
        dx, dy = self.sx.derivative(s_q), self.sy.derivative(s_q)
        return np.arctan2(dy, dx)

    def curvature(self, s_q: float | np.ndarray):
        ddx, ddy = self.sx.derivative2(s_q), self.sy.derivative2(s_q)
        dx, dy = self.sx.derivative(s_q), self.sy.derivative(s_q)
        num = dx * ddy - dy * ddx
        den = np.power(dx * dx + dy * dy, 1.5)
        den = np.where(den == 0, 1e-9, den)
        return num / den

    @property
    def length(self) -> float:
        return float(self.s[-1])


class _CubicSpline:
    """自然三次样条（1D）。"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        h = np.diff(x)
        a = y.copy()
        A = np.zeros((n, n))
        b = np.zeros(n)
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            b[i] = 3 * (a[i + 1] - a[i]) / h[i] - 3 * (a[i] - a[i - 1]) / h[i - 1]
        A[0, 0], A[-1, -1] = 1.0, 1.0
        c = np.linalg.solve(A, b)
        b_coef = np.zeros(n)
        d = np.zeros(n)
        for i in range(n - 1):
            b_coef[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3.0
            d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])
        self.x, self.a, self.b, self.c, self.d = x, a, b_coef, c, d

    def __call__(self, xq):
        xq = np.asarray(xq, dtype=float)
        idx = np.searchsorted(self.x, xq) - 1
        idx = np.clip(idx, 0, len(self.x) - 2)
        dx = xq - self.x[idx]
        return self.a[idx] + self.b[idx] * dx + self.c[idx] * dx ** 2 + self.d[idx] * dx ** 3

    def derivative(self, xq):
        xq = np.asarray(xq, dtype=float)
        idx = np.clip(np.searchsorted(self.x, xq) - 1, 0, len(self.x) - 2)
        dx = xq - self.x[idx]
        return self.b[idx] + 2 * self.c[idx] * dx + 3 * self.d[idx] * dx ** 2

    def derivative2(self, xq):
        xq = np.asarray(xq, dtype=float)
        idx = np.clip(np.searchsorted(self.x, xq) - 1, 0, len(self.x) - 2)
        dx = xq - self.x[idx]
        return 2 * self.c[idx] + 6 * self.d[idx] * dx


def make_double_lane_change(points: list[dict], ds: float = 0.5) -> Path:
    """双移线：给定路点 → 样条平滑 → 等弧长采样。"""
    cx = np.array([p["x"] for p in points], dtype=float)
    cy = np.array([p["y"] for p in points], dtype=float)
    spline = CubicSpline2D(cx, cy)
    s_q = np.arange(0.0, spline.length, ds)
    px, py = spline.position(s_q)
    yaw = spline.yaw(s_q)
    curv = spline.curvature(s_q)
    return Path(px, py, yaw, curv, s_q)


def make_circle(radius: float = 30.0, arc_deg: float = 270.0,
                center_x: float = 0.0, center_y: float = 30.0,
                ds: float = 0.5) -> Path:
    """环形路径：从 (center_x, center_y - radius) 起逆时针。"""
    total_len = np.deg2rad(arc_deg) * radius
    s = np.arange(0.0, total_len, ds)
    theta = s / radius  # 起始角 -90°
    px = center_x + radius * np.sin(theta)
    py = (center_y - radius) + radius * (1 - np.cos(theta))
    yaw = theta  # 切向
    curv = np.full_like(s, 1.0 / radius)
    return Path(px, py, yaw, curv, s)


def build_path(name: str, path_cfg: dict) -> Path:
    if name == "double_lane_change":
        return make_double_lane_change(path_cfg["segments"])
    if name == "circle":
        return make_circle(
            radius=float(path_cfg["radius"]),
            arc_deg=float(path_cfg["arc_deg"]),
            center_x=float(path_cfg["center_x"]),
            center_y=float(path_cfg["center_y"]),
        )
    raise ValueError(f"未知路径: {name}")


def nearest_index(path: Path, x: float, y: float, hint: int = 0, window: int = 200) -> int:
    """局部最近邻索引（hint 邻域搜索，避免全量）。"""
    n = len(path)
    lo = max(0, hint - window)
    hi = min(n, hint + window)
    xs, ys = path.x[lo:hi], path.y[lo:hi]
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    return int(lo + np.argmin(d2))

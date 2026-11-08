"""三类路径跟踪控制器：Pure Pursuit / Stanley / 非线性 MPC（CasADi）。

统一接口：compute(state, path, target_speed, ref_idx) -> (accel, steer, ref_idx, info)
与 PythonRobotics 参数量级对齐（k/Lfc/k/L/horizon/Q/R/Rd）。
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path as FsPath

import numpy as np

from .path import Path, nearest_index
from .vehicle import KinematicBicycle, VehicleState


def _ensure_casadi_dll_path():
    """Windows 下 casadi 插件 DLL 依赖需在 PATH 上（error 126 根因）。
    注意：不要用 os.add_dll_directory —— 它会把 DLL 搜索切到 LOAD_LIBRARY_SEARCH_*
    语义、反而找不到插件的同目录依赖。只改 PATH 即可。必须早于 import casadi。
    注意：本模块 from .path import Path 会遮蔽 pathlib.Path，故此处用 FsPath。"""
    try:
        import importlib.util
        spec = importlib.util.find_spec("casadi")
        if spec is None or spec.origin is None:
            return
        d = str(FsPath(spec.origin).resolve().parent)
        path = os.environ.get("PATH", "")
        if d not in path:
            os.environ["PATH"] = d + os.pathsep + path
    except Exception:
        pass


_ensure_casadi_dll_path()  # 模块加载即注入，必须早于任何 casadi import


# ---------------- Pure Pursuit ----------------

@dataclass
class PurePursuit:
    wheelbase: float = 2.9
    k: float = 0.1
    lfc: float = 2.0
    max_steer: float = 0.7854

    def compute(self, state: VehicleState, path: Path, target_speed: float,
                ref_idx: int, dt: float = 0.1):
        idx = nearest_index(path, state.x, state.y, hint=ref_idx)
        ld = max(self.lfc + self.k * state.v, 0.5)
        # 沿路径弧长找 lookahead 点
        s_ref = path.s[idx]
        s_target = s_ref + ld
        if s_target >= path.s[-1]:
            idx_t = len(path) - 1
        else:
            idx_t = int(np.searchsorted(path.s, s_target))
        tx, ty = path.x[idx_t], path.y[idx_t]
        alpha = np.arctan2(ty - state.y, tx - state.x) - state.yaw
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        steer = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), ld)
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        accel = 1.0 * (target_speed - state.v)  # 简单 P 速度环
        return accel, steer, idx, {"lookahead": ld, "alpha": alpha}


# ---------------- Stanley ----------------

@dataclass
class Stanley:
    k: float = 0.5
    k_soft: float = 1.0
    max_steer: float = 0.7854
    wheelbase: float = 2.9

    def compute(self, state: VehicleState, path: Path, target_speed: float,
                ref_idx: int, dt: float = 0.1):
        # 前轴位置（PythonRobotics: fx = x + L*cos(yaw)）
        fx = state.x + self.wheelbase * np.cos(state.yaw)
        fy = state.y + self.wheelbase * np.sin(state.yaw)
        idx = nearest_index(path, fx, fy, hint=ref_idx)
        # PythonRobotics 符号约定：error = (车-路)·[sin(yaw), -cos(yaw)]
        #   车在路径左侧 → error < 0 → 向右打舵
        dx, dy = fx - path.x[idx], fy - path.y[idx]
        e_front = dx * np.sin(state.yaw) - dy * np.cos(state.yaw)
        heading_err = path.yaw[idx] - state.yaw
        heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))
        steer = heading_err + np.arctan2(self.k * e_front, state.v + self.k_soft)
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        accel = 1.0 * (target_speed - state.v)
        return accel, steer, idx, {"e_front": e_front, "heading_err": heading_err}


# ---------------- MPC ----------------

@dataclass
class MPCController:
    wheelbase: float = 2.9
    horizon: int = 5
    q: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.5, 0.5])
    r: list[float] = field(default_factory=lambda: [0.01, 0.01])
    rd: list[float] = field(default_factory=lambda: [0.01, 1.0])
    max_steer: float = 0.7854
    max_steer_rate: float = 0.5236
    max_accel: float = 3.0
    max_decel: float = -5.0
    dt: float = 0.1

    _solver: object = field(default=None, repr=False)
    _last_u: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        _ensure_casadi_dll_path()
        self._build_solver()

    def _build_solver(self):
        import casadi as ca

        N = self.horizon
        dt, L = self.dt, self.wheelbase
        nx, nu = 4, 2  # [x,y,v,yaw], [accel, steer]

        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        v = ca.SX.sym("v")
        yaw = ca.SX.sym("yaw")
        states = ca.vertcat(x, y, v, yaw)
        n_states = states.numel()

        a = ca.SX.sym("a")
        delta = ca.SX.sym("delta")
        controls = ca.vertcat(a, delta)
        n_controls = controls.numel()

        rhs = ca.vertcat(v * ca.cos(yaw), v * ca.sin(yaw), a, v * ca.tan(delta) / L)
        f = ca.Function("f", [states, controls], [rhs])

        # 决策变量
        U = ca.SX.sym("U", n_controls, N)
        X = ca.SX.sym("X", n_states, N + 1)
        # P = [x0 (nx); x_ref_0..x_ref_N (nx*(N+1))] —— 逐步跟踪参考轨迹
        P = ca.SX.sym("P", n_states + n_states * (N + 1))

        obj = 0
        g = [X[:, 0] - P[:n_states]]

        Q = ca.diag(ca.DM(self.q))
        R = ca.diag(ca.DM(self.r))
        Rd = ca.diag(ca.DM(self.rd))

        for k in range(N):
            ref_k = P[n_states + k * n_states: n_states + (k + 1) * n_states]
            st = X[:, k]
            ct = U[:, k]
            obj += ca.mtimes([(st - ref_k).T, Q, (st - ref_k)])
            obj += ca.mtimes([ct.T, R, ct])
            if k > 0:
                du = U[:, k] - U[:, k - 1]
                obj += ca.mtimes([du.T, Rd, du])
            st_next = X[:, k + 1]
            k1 = f(st, ct)
            k2 = f(st + dt / 2 * k1, ct)
            k3 = f(st + dt / 2 * k2, ct)
            k4 = f(st + dt * k3, ct)
            st_pred = st + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            g.append(st_next - st_pred)

        ref_N = P[n_states + N * n_states: n_states + (N + 1) * n_states]
        obj += ca.mtimes([(X[:, N] - ref_N).T, Q * 5, (X[:, N] - ref_N)])

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        opt_params = P
        g_vec = ca.vertcat(*g)

        opts = {
            "ipopt.print_level": 0, "print_time": 0,
            "ipopt.max_iter": 100, "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3, "ipopt.mu_init": 0.1,
        }
        nlp = {"x": opt_vars, "f": obj, "g": g_vec, "p": opt_params}
        self._solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        self._nx, self._nu, self._N = n_states, n_controls, N
        self._lbx = []
        self._ubx = []
        for _ in range(N + 1):
            self._lbx += [-ca.inf] * nx
            self._ubx += [ca.inf] * nx
        for _ in range(N):
            self._lbx += [self.max_decel, -self.max_steer]
            self._ubx += [self.max_accel, self.max_steer]
        self._lbg = [0.0] * ((N + 1) * nx)
        self._ubg = [0.0] * ((N + 1) * nx)

    def _rollout(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        """用给定输入序列前向积分，生成动态可行的 X 初值 (nx, N+1)。"""
        N = self._N
        dt, L = self.dt, self.wheelbase
        X = np.zeros((self._nx, N + 1))
        X[:, 0] = x0
        s = x0.copy()
        for k in range(N):
            a, d = float(u_seq[0, k]), float(u_seq[1, k])
            x_, y_, v_, yaw_ = s
            k1 = np.array([v_ * np.cos(yaw_), v_ * np.sin(yaw_), v_ * np.tan(d) / L, a])
            s2 = s + dt / 2 * k1
            k2 = np.array([s2[2] * np.cos(s2[3]), s2[2] * np.sin(s2[3]),
                           s2[2] * np.tan(d) / L, a])
            s3 = s + dt / 2 * k2
            k3 = np.array([s3[2] * np.cos(s3[3]), s3[2] * np.sin(s3[3]),
                           s3[2] * np.tan(d) / L, a])
            s4 = s + dt * k3
            k4 = np.array([s4[2] * np.cos(s4[3]), s4[2] * np.sin(s4[3]),
                           s4[2] * np.tan(d) / L, a])
            s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            X[:, k + 1] = s
        return X

    def compute(self, state: VehicleState, path: Path, target_speed: float,
                ref_idx: int, dt: float = 0.1):
        t0 = time.perf_counter()
        N = self._N
        nx, nu = self._nx, self._nu
        # 先取最近点索引，再以其弧长为参考起点（不能直接用 ref_idx，那只是 hint）
        idx = nearest_index(path, state.x, state.y, hint=ref_idx)
        s0 = path.s[idx]
        ds = max(state.v, 1.0) * dt
        x_ref = np.zeros((nx, N + 1))
        for i in range(N + 1):
            s_q = min(s0 + i * ds, path.s[-1])
            j = int(np.searchsorted(path.s, s_q))
            j = min(j, len(path) - 1)
            x_ref[0, i] = path.x[j]
            x_ref[1, i] = path.y[j]
            x_ref[2, i] = target_speed
            x_ref[3, i] = path.yaw[j]

        x0 = np.array([state.x, state.y, state.v, state.yaw])
        # CasADi reshape 为列主序：须按 time-major 展平 (x_k,y_k,v_k,yaw_k)×(N+1)
        p = np.concatenate([x0, x_ref.flatten(order="F")])

        # 初值：U 用上次解（或 0），X 用前向 rollout（动态可行）
        u_init = np.zeros((nu, N)) if self._last_u is None else np.tile(
            self._last_u.reshape(-1, 1), (1, N))
        u_init[0] = np.clip(u_init[0], self.max_decel, self.max_accel)
        u_init[1] = np.clip(u_init[1], -self.max_steer, self.max_steer)
        x_init = self._rollout(x0, u_init)
        w0 = np.concatenate([x_init.flatten(order="F"), u_init.flatten(order="F")])

        sol = self._solver(x0=w0, p=p, lbx=self._lbx, ubx=self._ubx,
                           lbg=self._lbg, ubg=self._ubg)
        w = np.array(sol["x"]).flatten()
        u0 = w[(N + 1) * nx:(N + 1) * nx + nu].copy()  # 列主序下前 nu 个即 u_0
        u0[0] = float(np.clip(u0[0], self.max_decel, self.max_accel))
        u0[1] = float(np.clip(u0[1], -self.max_steer, self.max_steer))
        self._last_u = u0.copy()

        idx = nearest_index(path, state.x, state.y, hint=ref_idx)
        solve_ms = (time.perf_counter() - t0) * 1000.0
        return float(u0[0]), float(u0[1]), idx, {"solve_ms": solve_ms}


def make_controller(name: str, cfg: dict, vehicle_cfg: dict):
    """工厂：name ∈ {pure_pursuit, stanley, mpc}。"""
    max_steer = float(vehicle_cfg["max_steer"])
    wb = float(vehicle_cfg["wheelbase"])
    dt = float(vehicle_cfg["dt"])
    if name == "pure_pursuit":
        p = cfg["pure_pursuit"]
        return PurePursuit(wheelbase=wb, k=float(p["k"]), lfc=float(p["lfc"]),
                           max_steer=max_steer), "PurePursuit"
    if name == "stanley":
        p = cfg["stanley"]
        return Stanley(k=float(p["k"]), k_soft=float(p["k_soft"]),
                       max_steer=max_steer, wheelbase=wb), "Stanley"
    if name == "mpc":
        p = cfg["mpc"]
        return MPCController(
            wheelbase=wb, horizon=int(p["horizon"]),
            q=list(p["q"]), r=list(p["r"]), rd=list(p["rd"]),
            max_steer=max_steer,
            max_steer_rate=float(vehicle_cfg["max_steer_rate"]),
            max_accel=float(p["max_accel"]), max_decel=float(p["max_decel"]),
            dt=dt,
        ), "MPC"
    raise ValueError(f"未知控制器: {name}")

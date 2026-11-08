"""评价指标：横向误差 / 航向误差 / 转角平滑度 / 计算耗时。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_track_error(x: np.ndarray, y: np.ndarray, yaw: np.ndarray,
                      ref_x: np.ndarray, ref_y: np.ndarray, ref_yaw: np.ndarray) -> np.ndarray:
    """带符号横向误差：投影到参考法向。>0 表示在参考左侧。"""
    dx = x - ref_x
    dy = y - ref_y
    # 法向 n = [-sin(yaw_ref), cos(yaw_ref)]
    return -dx * np.sin(ref_yaw) + dy * np.cos(ref_yaw)


def heading_error(yaw: np.ndarray, ref_yaw: np.ndarray) -> np.ndarray:
    """航向误差（wrap 到 [-pi, pi]）。"""
    e = yaw - ref_yaw
    return np.arctan2(np.sin(e), np.cos(e))


def _rms(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")


def metric_pack(steer: np.ndarray, accel: np.ndarray, dt: float,
                e_lat: np.ndarray, e_yaw: np.ndarray,
                solve_times: np.ndarray | None = None) -> dict:
    """汇总一份轨迹的指标。"""
    steer = np.asarray(steer, dtype=float)
    accel = np.asarray(accel, dtype=float)
    d_steer = np.diff(steer) / dt if len(steer) > 1 else np.array([0.0])
    dd_steer = np.diff(d_steer) / dt if len(d_steer) > 1 else np.array([0.0])
    out = {
        "lat_rms": _rms(e_lat),
        "lat_max": float(np.max(np.abs(e_lat))) if len(e_lat) else float("nan"),
        "yaw_rms": _rms(e_yaw),
        "yaw_max": float(np.max(np.abs(e_yaw))) if len(e_yaw) else float("nan"),
        "steer_rate_rms": _rms(d_steer),
        "steer_jerk_rms": _rms(dd_steer),
        "steer_max": float(np.max(np.abs(steer))) if len(steer) else float("nan"),
        "accel_rms": _rms(accel),
        "n_steps": int(len(steer)),
    }
    if solve_times is not None and len(solve_times):
        st = np.asarray(solve_times, dtype=float) * 1000.0
        out["solve_ms_mean"] = float(np.mean(st))
        out["solve_ms_p95"] = float(np.percentile(st, 95))
    return out


def trajectory_frame(t: np.ndarray, x, y, yaw, v, steer, accel,
                     e_lat, e_yaw) -> pd.DataFrame:
    return pd.DataFrame({
        "t": t, "x": x, "y": y, "yaw": yaw, "v": v,
        "steer": steer, "accel": accel, "e_lat": e_lat, "e_yaw": e_yaw,
    })

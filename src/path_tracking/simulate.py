"""闭环仿真循环（v1.1：可选动力学植物 / 侧风 / 量测噪声）。"""

from __future__ import annotations

import numpy as np

from .metrics import cross_track_error, heading_error, metric_pack, trajectory_frame
from .path import Path, nearest_index
from .vehicle import DynamicBicycle, KinematicBicycle, VehicleState


def simulate(
    controller,
    path: Path,
    vehicle_cfg: dict,
    target_speed: float,
    init: dict | None = None,
    settle_s: float = 2.0,
    max_time: float | None = None,
    plant: str = "kinematic",
    wind_accel: float = 0.0,
    meas_noise_std: float = 0.0,
    noise_seed: int = 0,
    dyn_cfg: dict | None = None,
) -> dict:
    """闭环仿真直到跑完路径或超时。

    plant: "kinematic" | "dynamic"（2-DOF 线性轮胎动力学自行车）
    wind_accel: 侧风等效横向加速度 [m/s^2]（动力学植物）
    meas_noise_std: 控制器量测噪声（x/y/yaw/v 高斯白噪声标准差，yaw 为 rad）
    返回 dict: frame(DataFrame), metrics(dict), meta(dict)
    """
    dt = float(vehicle_cfg["dt"])
    if plant == "dynamic":
        dyn_cfg = dyn_cfg or {}
        veh = DynamicBicycle(
            wheelbase=float(vehicle_cfg["wheelbase"]),
            max_steer=float(vehicle_cfg["max_steer"]),
            max_steer_rate=float(vehicle_cfg["max_steer_rate"]),
            dt=dt,
            mass=float(dyn_cfg.get("mass", 1573.0)),
            iz=float(dyn_cfg.get("iz", 2500.0)),
            lf=float(dyn_cfg.get("lf", 1.2)),
            lr=float(dyn_cfg.get("lr", 1.7)),
            cf=float(dyn_cfg.get("cf", 80000.0)),
            cr=float(dyn_cfg.get("cr", 80000.0)),
            wind_accel=wind_accel,
        )
    else:
        veh = KinematicBicycle(
            wheelbase=float(vehicle_cfg["wheelbase"]),
            max_steer=float(vehicle_cfg["max_steer"]),
            max_steer_rate=float(vehicle_cfg["max_steer_rate"]),
            dt=dt,
        )
    init = init or {}
    veh.reset(init.get("x", path.x[0]), init.get("y", path.y[0]),
              init.get("yaw", path.yaw[0]), init.get("v", target_speed))

    if max_time is None:
        max_time = (path.s[-1] / max(target_speed, 1.0)) * 1.6 + 5.0

    n_max = int(max_time / dt)
    rec = {k: [] for k in ("t", "x", "y", "yaw", "v", "steer", "accel", "solve_ms")}
    ref_idx = 0
    rng = np.random.default_rng(noise_seed)

    for i in range(n_max):
        s_true = veh.state
        if meas_noise_std > 0.0:
            obs = VehicleState(
                s_true.x + rng.normal(0, meas_noise_std),
                s_true.y + rng.normal(0, meas_noise_std),
                s_true.yaw + rng.normal(0, meas_noise_std),
                s_true.v + rng.normal(0, meas_noise_std),
            )
        else:
            obs = s_true
        accel, steer, ref_idx, info = controller.compute(obs, path, target_speed, ref_idx, dt)
        s = veh.step(accel, steer, dt)
        rec["t"].append(i * dt)
        rec["x"].append(s.x)
        rec["y"].append(s.y)
        rec["yaw"].append(s.yaw)
        rec["v"].append(s.v)
        rec["steer"].append(steer)
        rec["accel"].append(accel)
        rec["solve_ms"].append(info.get("solve_ms", 0.0) / 1000.0)  # 存秒，metric_pack 转 ms
        # 终止：弧长走完
        if ref_idx >= len(path) - 2:
            break

    # 对齐参考
    n = len(rec["t"])
    ref_x = np.zeros(n)
    ref_y = np.zeros(n)
    ref_yaw = np.zeros(n)
    hint = 0
    for i in range(n):
        hint = nearest_index(path, rec["x"][i], rec["y"][i], hint=hint, window=80)
        ref_x[i] = path.x[hint]
        ref_y[i] = path.y[hint]
        ref_yaw[i] = path.yaw[hint]

    e_lat = cross_track_error(np.array(rec["x"]), np.array(rec["y"]), np.array(rec["yaw"]),
                              ref_x, ref_y, ref_yaw)
    e_yaw = heading_error(np.array(rec["yaw"]), ref_yaw)

    # 起始对齐段剔除
    skip = int(settle_s / dt)
    sl = slice(skip, None)

    m = metric_pack(
        np.array(rec["steer"])[sl], np.array(rec["accel"])[sl], dt,
        e_lat[sl], e_yaw[sl],
        solve_times=np.array(rec["solve_ms"])[sl] if any(rec["solve_ms"]) else None,
    )
    frame = trajectory_frame(
        np.array(rec["t"]), np.array(rec["x"]), np.array(rec["y"]), np.array(rec["yaw"]),
        np.array(rec["v"]), np.array(rec["steer"]), np.array(rec["accel"]),
        e_lat, e_yaw,
    )
    return {
        "frame": frame,
        "metrics": m,
        "ref_x": ref_x, "ref_y": ref_y, "ref_yaw": ref_yaw,
        "e_lat": e_lat, "e_yaw": e_yaw,
        "target_speed": target_speed,
    }

"""03 深化实验（v1.1）：动力学植物 / 自适应 horizon / 曲率前馈 / 变曲率路径 / 鲁棒性。

实验矩阵：
  E1 动力学自行车植物：三控制器（运动学调参）在 2-DOF 模型上的迁移性
  E2 速度自适应 horizon MPC-A：验证「速度反转」是否被消除
  E3 Stanley 曲率前馈：恒曲率稳态误差是否消除
  E4 变曲率 clothoid S 弯：连续变曲率跟踪
  E5 鲁棒性：侧风扰动 / 量测噪声
  E6 终端代价 × horizon 消融：速度反转根因（负结果：加长视野无效，终端代价有害）
输出：results/deepening_metrics.json, results/deepening_rows.csv, figures/deepening_*.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from path_tracking import plots
from path_tracking.config import default_config_path, load_config
from path_tracking.controllers import make_controller
from path_tracking.path import build_path
from path_tracking.simulate import simulate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CTRLS = ("pure_pursuit", "stanley", "mpc")


def run_one(cfg, path, path_name, speed_name, speed, ctrl_key, tag,
            plant="kinematic", wind_accel=0.0, meas_noise_std=0.0,
            dyn_cfg=None, settle_s=2.0, ctrl_overrides=None):
    veh_cfg = cfg["vehicle"]
    controller, label = make_controller(ctrl_key, cfg["controllers"], veh_cfg)
    if ctrl_overrides:
        for k, v in ctrl_overrides.items():
            setattr(controller, k, v)
        label = label + "+" + ",".join(f"{k}={v}" for k, v in ctrl_overrides.items())
    t0 = time.perf_counter()
    out = simulate(
        controller, path, veh_cfg, speed,
        init={"x": path.x[0], "y": path.y[0], "yaw": path.yaw[0], "v": speed},
        settle_s=settle_s,
        plant=plant, wind_accel=wind_accel,
        meas_noise_std=meas_noise_std,
        dyn_cfg=dyn_cfg,
    )
    wall = time.perf_counter() - t0
    m = out["metrics"]
    m.update({
        "wall_s": wall, "scenario": tag, "model": label,
        "path": path_name, "speed_name": speed_name, "speed_mps": speed,
        "plant": plant, "wind_accel": wind_accel, "meas_noise_std": meas_noise_std,
    })
    print(f"[{tag}] {label:11s} lat_rms={m['lat_rms']:.3f}m lat_max={m['lat_max']:.2f}m "
          f"yaw_rms={m['yaw_rms']:.4f} steer_rate={m['steer_rate_rms']:.4f} "
          f"solve_ms={m.get('solve_ms_mean', 0):.1f} wall={wall:.1f}s")
    return m, out


def main():
    cfg = load_config(default_config_path())
    veh_cfg = cfg["vehicle"]
    speed_cfg = cfg["speed_levels"]
    path_cfg = cfg["paths"]
    dyn_cfg = cfg.get("dynamic_vehicle", {})
    settle_s = float(cfg["sim"]["settle_s"])

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    rows = []

    # ---------- E1 动力学植物迁移 ----------
    print("\n===== E1 动力学自行车植物（2-DOF 线性轮胎）=====")
    e1_details = {}
    for path_name in ("double_lane_change", "circle"):
        path = build_path(path_name, path_cfg[path_name])
        for speed_name, speed in speed_cfg.items():
            speed = float(speed)
            tag = f"dyn_{path_name}_{speed_name}"
            results = {}
            for ctrl_key in CTRLS:
                m, out = run_one(cfg, path, path_name, speed_name, speed,
                                 ctrl_key, tag, plant="dynamic", dyn_cfg=dyn_cfg,
                                 settle_s=settle_s)
                rows.append(m)
                results[m["model"]] = out
            e1_details[tag] = results
            plots.plot_trajectory(path, results, f"动力学植物 · {tag}",
                                  FIGURES / f"{tag}_traj.png")
            plots.plot_errors(results, f"动力学植物误差 · {tag}",
                              FIGURES / f"{tag}_error.png")

    # ---------- E2 速度自适应 horizon（MPC vs MPC-A）----------
    print("\n===== E2 速度自适应 horizon（速度反转验证）=====")
    e2_details = {}
    for path_name in ("double_lane_change", "circle"):
        path = build_path(path_name, path_cfg[path_name])
        for speed_name, speed in speed_cfg.items():
            speed = float(speed)
            tag = f"adapt_{path_name}_{speed_name}"
            results = {}
            for ctrl_key in ("mpc", "mpc_adaptive", "pure_pursuit"):
                m, out = run_one(cfg, path, path_name, speed_name, speed,
                                 ctrl_key, tag, settle_s=settle_s)
                rows.append(m)
                results[m["model"]] = out
            e2_details[tag] = results
            plots.plot_trajectory(path, results, f"自适应 horizon · {tag}",
                                  FIGURES / f"{tag}_traj.png")
            plots.plot_errors(results, f"自适应 horizon 误差 · {tag}",
                              FIGURES / f"{tag}_error.png")

    # ---------- E3 Stanley 曲率前馈 ----------
    print("\n===== E3 Stanley 曲率前馈（恒曲率稳态误差）=====")
    e3_details = {}
    for path_name in ("circle", "double_lane_change"):
        path = build_path(path_name, path_cfg[path_name])
        for speed_name, speed in speed_cfg.items():
            speed = float(speed)
            tag = f"ff_{path_name}_{speed_name}"
            results = {}
            for ctrl_key in ("stanley", "stanley_ff"):
                m, out = run_one(cfg, path, path_name, speed_name, speed,
                                 ctrl_key, tag, settle_s=settle_s)
                rows.append(m)
                results[m["model"]] = out
            e3_details[tag] = results
            plots.plot_errors(results, f"曲率前馈误差 · {tag}",
                              FIGURES / f"{tag}_error.png")

    # ---------- E4 变曲率 clothoid S 弯 ----------
    print("\n===== E4 变曲率 clothoid S 弯 =====")
    path = build_path("clothoid_s", path_cfg["clothoid_s"])
    e4_details = {}
    for speed_name, speed in speed_cfg.items():
        speed = float(speed)
        tag = f"clothoid_{speed_name}"
        results = {}
        for ctrl_key in CTRLS:
            m, out = run_one(cfg, path, "clothoid_s", speed_name, speed,
                             ctrl_key, tag, settle_s=settle_s)
            rows.append(m)
            results[m["model"]] = out
        e4_details[tag] = results
        plots.plot_trajectory(path, results, f"变曲率 S 弯 · {tag}",
                              FIGURES / f"{tag}_traj.png")
        plots.plot_errors(results, f"变曲率 S 弯误差 · {tag}",
                          FIGURES / f"{tag}_error.png")

    # ---------- E5 鲁棒性：侧风 / 量测噪声 ----------
    print("\n===== E5 鲁棒性（侧风 0.5 m/s² / 量测噪声 σ=5cm）=====")
    path = build_path("double_lane_change", path_cfg["double_lane_change"])
    e5_specs = [
        ("wind", dict(plant="dynamic", dyn_cfg=dyn_cfg, wind_accel=0.5)),
        ("noise", dict(meas_noise_std=0.05)),
    ]
    e5_details = {}
    for rname, rkw in e5_specs:
        for speed_name, speed in speed_cfg.items():
            speed = float(speed)
            tag = f"robust_{rname}_{speed_name}"
            results = {}
            for ctrl_key in CTRLS:
                m, out = run_one(cfg, path, "double_lane_change", speed_name, speed,
                                 ctrl_key, tag, settle_s=settle_s, **rkw)
                rows.append(m)
                results[m["model"]] = out
            e5_details[tag] = results
            plots.plot_errors(results, f"鲁棒性({rname})误差 · {tag}",
                              FIGURES / f"{tag}_error.png")

    # ---------- E6 终端代价 × horizon 消融（速度反转根因）----------
    # 诊断结论（DLC 高速）：N≥10 全面崩坏；N=5 时终端代价单调有害（w=0 最优）；
    # 但最优 MPC 配置下高速 PP 仍更优 —— 「加长视野修复速度反转」假说被证伪。
    print("\n===== E6 终端代价 × horizon 消融（DLC 高速）=====")
    path = build_path("double_lane_change", path_cfg["double_lane_change"])
    speed = float(speed_cfg["high"])
    e6_sweep = []
    for n in (5, 10, 15, 20):
        for tw in (0.0, 1.0, 5.0):
            m, _ = run_one(cfg, path, "double_lane_change", "high", speed, "mpc",
                           "ablate_dlc_high", settle_s=settle_s,
                           ctrl_overrides={"horizon": n, "term_w": tw})
            m["horizon_N"] = n
            m["term_w"] = tw
            rows.append(m)
            e6_sweep.append(m)

    # 最优 MPC 配置（N=5, term_w=0）vs 经典三控制器：速度反转是否消除？
    print("\n===== E6b 最优 MPC 配置 vs PP/Stanley（速度反转验证）=====")
    e6b_details = {}
    for path_name in ("double_lane_change", "circle"):
        path = build_path(path_name, path_cfg[path_name])
        for speed_name, speed in speed_cfg.items():
            speed = float(speed)
            tag = f"best_{path_name}_{speed_name}"
            results = {}
            specs = [("mpc", {"term_w": 5.0}), ("mpc", {"term_w": 0.0}),
                     ("pure_pursuit", None), ("stanley_ff", None)]
            for ctrl_key, ov in specs:
                m, out = run_one(cfg, path, path_name, speed_name, speed,
                                 ctrl_key, tag, settle_s=settle_s, ctrl_overrides=ov)
                rows.append(m)
                results[m["model"]] = out
            e6b_details[tag] = results
            plots.plot_errors(results, f"最优 MPC 对照误差 · {tag}",
                              FIGURES / f"{tag}_error.png")

    # ---------- 汇总 ----------
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "deepening_rows.csv", index=False, encoding="utf-8-sig")

    bar_rows = [{"scenario": r["scenario"], "model": r["model"], "value": r["lat_rms"]}
                for r in rows]
    plots.plot_metric_bars(bar_rows, "lat_rms", "横向误差 RMS [m]",
                           "深化实验跟踪精度对比", FIGURES / "deepening_bar_lat_rms.png")

    summary = {"rows": rows}
    (RESULTS / "deepening_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print(f"\n共 {len(rows)} 条指标 → {RESULTS}")
    for r in rows:
        if not np.isfinite(r["lat_rms"]):
            print(f"WARN: {r['scenario']}/{r['model']} lat_rms 非有限")
        if r["lat_max"] > 5.0:
            print(f"WARN: {r['scenario']}/{r['model']} lat_max={r['lat_max']:.1f}m 偏大")


if __name__ == "__main__":
    main()

"""01 主对比实验：路径 × 速度 × 控制器 → 指标表 + 图。

场景矩阵：
  path ∈ {double_lane_change, circle}
  speed ∈ {low (20 km/h), high (60 km/h)}
  controller ∈ {PurePursuit, Stanley, MPC}
输出：results/comparison_metrics.json, results/comparison_rows.csv, figures/*.png
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


def main():
    cfg = load_config(default_config_path())
    veh_cfg = cfg["vehicle"]
    ctrl_cfg = cfg["controllers"]
    speed_cfg = cfg["speed_levels"]
    path_cfg = cfg["paths"]

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    rows = []
    details = {}

    for path_name in ("double_lane_change", "circle"):
        path = build_path(path_name, path_cfg[path_name])
        for speed_name, speed in speed_cfg.items():
            speed = float(speed)
            key = f"{path_name}_{speed_name}"
            results = {}
            for ctrl_key in ("pure_pursuit", "stanley", "mpc"):
                controller, label = make_controller(ctrl_key, ctrl_cfg, veh_cfg)
                t0 = time.perf_counter()
                out = simulate(controller, path, veh_cfg, speed,
                               init={"x": path.x[0], "y": path.y[0], "yaw": path.yaw[0], "v": speed},
                               settle_s=float(cfg["sim"]["settle_s"]))
                wall = time.perf_counter() - t0
                m = out["metrics"]
                m["wall_s"] = wall
                m["scenario"] = key
                m["model"] = label
                m["path"] = path_name
                m["speed_name"] = speed_name
                m["speed_mps"] = speed
                rows.append(m)
                results[label] = out
                print(f"[{key}] {label:11s} lat_rms={m['lat_rms']:.3f}m lat_max={m['lat_max']:.2f}m "
                      f"yaw_rms={m['yaw_rms']:.4f} steer_rate={m['steer_rate_rms']:.4f} "
                      f"solve_ms={m.get('solve_ms_mean', 0):.1f} wall={wall:.1f}s")
            details[key] = results

            tag = f"{path_name}_{speed_name}"
            plots.plot_trajectory(path, results, f"轨迹叠加 · {tag}", FIGURES / f"{tag}_traj.png")
            plots.plot_errors(results, f"误差曲线 · {tag}", FIGURES / f"{tag}_error.png")
            plots.plot_steer(results, f"控制量 · {tag}", FIGURES / f"{tag}_steer.png")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "comparison_rows.csv", index=False, encoding="utf-8-sig")

    # 条形图：lat_rms / steer_rate / solve_ms
    for metric, ylabel, title in [
        ("lat_rms", "横向误差 RMS [m]", "跟踪精度对比"),
        ("steer_rate_rms", "转角速率 RMS [rad/s]", "控制平滑度对比"),
        ("yaw_rms", "航向误差 RMS [rad]", "航向精度对比"),
    ]:
        bar_rows = [{"scenario": r["scenario"], "model": r["model"], "value": r[metric]}
                    for r in rows]
        plots.plot_metric_bars(bar_rows, metric, ylabel, title,
                               FIGURES / f"bar_{metric}.png")

    summary = {"rows": rows}
    (RESULTS / "comparison_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print(f"\n共 {len(rows)} 条指标，图 {len(list(FIGURES.glob('*.png')))} 张 → {RESULTS}")
    # 快速健康检查
    for r in rows:
        if not np.isfinite(r["lat_rms"]):
            print(f"WARN: {r['scenario']}/{r['model']} lat_rms 非有限")
        if r["lat_max"] > 5.0:
            print(f"WARN: {r['scenario']}/{r['model']} lat_max={r['lat_max']:.1f}m 偏大")


if __name__ == "__main__":
    main()

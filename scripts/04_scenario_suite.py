"""04 场景套件（v1.2）：E7 场景族 / E8 Stanley 速度调度增益 / E9 失效包络。

  E7 场景族：双移线（横移 2.0/3.5/5.0 m）与 S 弯（κ_max 0.02/0.05/0.10）× 20/60 km/h
           —— "低速 MPC / 高速 PP" 的选型结论在场景族上是否稳定？
  E8 Stanley 速度调度：const vs linear vs speed_inv 增益，跨 20/40/60 km/h
           —— 调度能否压平 Stanley 的速度退化？
  E9 失效包络：量测噪声 {0, 5, 10, 20 cm} × 速度 {20, 40, 60} × 环形 R {15, 30, 60} m
           —— 每个控制器的失效边界（lat_rms > 0.3 m 判失败）。
输出：results/scenario_suite.json、figures/fig21–23_*.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from path_tracking import plots  # noqa: F401  字体 rcParams 副作用
from path_tracking.config import default_config_path, load_config
from path_tracking.controllers import make_controller
from path_tracking.path import (
    Path as RefPath,  # 勿遮蔽 pathlib.Path（v1.1 坑）
    make_circle,
    make_clothoid_s_curve,
    make_double_lane_change,
)
from path_tracking.simulate import simulate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FAIL_LAT_RMS = 0.3  # m
SPEEDS = {"20": 5.5556, "40": 11.1111, "60": 16.6667}
# ctrl_key → make_controller 返回的 label（绘图匹配用，勿用 `c in kk` 小写匹配）
LABEL_OF = {"pure_pursuit": "PurePursuit", "stanley": "Stanley", "mpc": "MPC"}


def run_one(cfg, path, tag, speed, ctrl_key, plant="kinematic",
            meas_noise_std=0.0, ctrl_overrides=None, settle_s=2.0):
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
        settle_s=settle_s, plant=plant, meas_noise_std=meas_noise_std,
    )
    m = out["metrics"]
    m.update({"scenario": tag, "model": label, "speed_mps": speed,
              "meas_noise_std": meas_noise_std,
              "wall_s": time.perf_counter() - t0,
              "fail": bool(m["lat_rms"] > FAIL_LAT_RMS)})
    print(f"[{tag}] {label:22s} lat_rms={m['lat_rms']:.3f}m fail={m['fail']}")
    return m


def dlc_family(offset: float) -> RefPath:
    """ISO 双移线按横移 offset 缩放。"""
    pts = [
        {"x": 0.0, "y": 0.0}, {"x": 35.0, "y": 0.0},
        {"x": 50.0, "y": offset}, {"x": 75.0, "y": offset},
        {"x": 90.0, "y": 0.0}, {"x": 130.0, "y": 0.0},
    ]
    return make_double_lane_change(pts)


def main():
    cfg = load_config(default_config_path())
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    out_all: dict = {"E7_scenarios": {}, "E8_schedule": {}, "E9_envelope": {}}
    rows = []

    # ---------- E7 场景族 ----------
    print("\n===== E7 场景族：双移线横移 / S 弯曲率 × 速度 =====")
    scenarios = {
        "dlc_off2.0": dlc_family(2.0),
        "dlc_off3.5": dlc_family(3.5),
        "dlc_off5.0": dlc_family(5.0),
        "s_k0.02": make_clothoid_s_curve(kappa_max=0.02),
        "s_k0.05": make_clothoid_s_curve(kappa_max=0.05),
        "s_k0.10": make_clothoid_s_curve(kappa_max=0.10),
    }
    for scen_name, path in scenarios.items():
        out_all["E7_scenarios"][scen_name] = {}
        for speed_name, speed in SPEEDS.items():
            key = f"{scen_name}_{speed_name}kmh"
            res = {}
            for ctrl_key in ("pure_pursuit", "stanley", "mpc"):
                m = run_one(cfg, path, key, speed, ctrl_key)
                rows.append(m)
                res[m["model"]] = {"lat_rms": m["lat_rms"], "lat_max": m["lat_max"],
                                   "fail": m["fail"]}
            out_all["E7_scenarios"][scen_name][speed_name] = res

    # 排名稳定性：每场景每速度的第一名
    win_count: dict = {}
    for scen, by_speed in out_all["E7_scenarios"].items():
        for sp, res in by_speed.items():
            best = min(res, key=lambda k: res[k]["lat_rms"])
            win_count[best] = win_count.get(best, 0) + 1
    out_all["E7_win_count"] = win_count
    print("E7 胜场:", win_count)

    # ---------- E8 Stanley 速度调度 ----------
    print("\n===== E8 Stanley 增益调度：const / linear / speed_inv =====")
    paths_e8 = {
        "circle": make_circle(radius=30.0, arc_deg=270.0),
        "dlc": dlc_family(3.5),
    }
    for path_name, path in paths_e8.items():
        out_all["E8_schedule"][path_name] = {}
        for speed_name, speed in SPEEDS.items():
            entry = {}
            for mode in ("const", "linear", "speed_inv"):
                m = run_one(cfg, path, f"e8_{path_name}_{speed_name}", speed,
                            "stanley", ctrl_overrides={"k_schedule": mode})
                rows.append(m)
                entry[mode] = {"lat_rms": m["lat_rms"], "lat_max": m["lat_max"]}
            out_all["E8_schedule"][path_name][speed_name] = entry
            print(f"  {path_name} {speed_name}kmh: " +
                  " ".join(f"{k}={v['lat_rms']:.3f}" for k, v in entry.items()))

    # ---------- E9 失效包络 ----------
    print("\n===== E9 失效包络：噪声 × 速度 × 曲率 =====")
    noises = [0.0, 0.05, 0.10, 0.20]
    radii = {"R15": 15.0, "R30": 30.0, "R60": 60.0}
    for r_name, radius in radii.items():
        path = make_circle(radius=radius, arc_deg=270.0)
        out_all["E9_envelope"][r_name] = {}
        for noise in noises:
            n_key = f"n{int(noise * 100):02d}"
            out_all["E9_envelope"][r_name][n_key] = {}
            for speed_name, speed in SPEEDS.items():
                res = {}
                for ctrl_key in ("pure_pursuit", "stanley", "mpc"):
                    m = run_one(cfg, path, f"e9_{r_name}_{n_key}_{speed_name}",
                                speed, ctrl_key, meas_noise_std=noise)
                    rows.append(m)
                    res[m["model"]] = {"lat_rms": m["lat_rms"], "fail": m["fail"]}
                out_all["E9_envelope"][r_name][n_key][speed_name] = res

    with open(RESULTS / "scenario_suite.json", "w", encoding="utf-8") as f:
        json.dump(out_all, f, ensure_ascii=False, indent=1)
    pd.DataFrame(rows).to_csv(RESULTS / "scenario_suite_rows.csv", index=False)

    # ---------- 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig21：E7 场景族 lat_rms 矩阵
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ctrls = ["pure_pursuit", "stanley", "mpc"]
    for ax, kind in zip(axes, ["dlc", "s"]):
        names = [n for n in scenarios if n.startswith(kind)]
        xs = np.arange(len(names))
        for k, c in enumerate(ctrls):
            vals = []
            for n in names:
                res = out_all["E7_scenarios"][n]["60"]
                vals.append(next(v["lat_rms"] for kk, v in res.items()
                                if kk.startswith(LABEL_OF[c])))
            ax.bar(xs + (k - 1) * 0.27, vals, 0.27, label=c)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=15, fontsize=7)
        ax.set_title(f"E7 {'双移线族' if kind=='dlc' else 'S弯族'}（60 km/h）")
        ax.set_ylabel("lat_rms [m]")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig21_scenario_suite.png", dpi=160, bbox_inches="tight")

    # fig22：E8 调度曲线
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, path_name in zip(axes, ["circle", "dlc"]):
        for mode in ("const", "linear", "speed_inv"):
            vals = [out_all["E8_schedule"][path_name][sp][mode]["lat_rms"]
                    for sp in ("20", "40", "60")]
            ax.plot([20, 40, 60], vals, "o-", label=mode)
        ax.set_title(f"E8 Stanley 调度 · {path_name}")
        ax.set_xlabel("速度 [km/h]")
        ax.set_ylabel("lat_rms [m]")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig22_stanley_schedule.png", dpi=160, bbox_inches="tight")

    # fig23：E9 失效包络（噪声 × 速度，每控制器一行，列=R）
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
    for i, c in enumerate(ctrls):
        for j, r_name in enumerate(radii):
            ax = axes[i][j]
            grid = np.zeros((len(noises), len(SPEEDS)))
            for a, noise in enumerate(noises):
                n_key = f"n{int(noise * 100):02d}"
                for b, sp in enumerate(("20", "40", "60")):
                    res = out_all["E9_envelope"][r_name][n_key][sp]
                    entry = next(v for kk, v in res.items()
                                 if kk.startswith(LABEL_OF[c]))
                    grid[a, b] = entry["lat_rms"]
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn_r",
                           vmin=0, vmax=2.0)
            for a in range(len(noises)):
                for b in range(len(SPEEDS)):
                    ok = grid[a, b] <= FAIL_LAT_RMS
                    ax.text(b, a, "O" if ok else "X", ha="center", va="center",
                            fontsize=10, color="black")
            ax.set_xticks(range(len(SPEEDS)))
            ax.set_xticklabels(("20", "40", "60"), fontsize=7)
            ax.set_yticks(range(len(noises)))
            ax.set_yticklabels([f"{n*100:.0f}cm" for n in noises], fontsize=7)
            if i == 0:
                ax.set_title(r_name, fontsize=9)
            if j == 0:
                ax.set_ylabel(c, fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.6, label="lat_rms [m]")
    fig.suptitle("E9 失效包络：O = lat_rms ≤ 0.3 m（通过）", fontsize=11)
    fig.savefig(FIGURES / "fig23_failure_envelope.png", dpi=150, bbox_inches="tight")

    print("图 →", FIGURES / "fig21_scenario_suite.png / fig22_stanley_schedule.png / fig23_failure_envelope.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""07 轨迹级速度规划（v1.3，E28）。

回应报告 §4.5："轨迹级速度规划"——由 E9 失效包络反推 v_safe(R, σ)，
沿路径生成速度剖面，证明规划后失效格大幅减少。

对比：
  fixed   —— 恒速（v1.2 E9 基线）
  planned —— 安全速度剖面（曲率自适应 + 噪声衰减 + 加减速平滑）

场景：环形 R ∈ {15, 30, 60} × σ ∈ {0, 5, 10} cm × 3 控制器
判据：lat_rms ≤ 0.3 m（与 E9 一致）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from path_tracking import plots  # noqa: F401
from path_tracking.config import default_config_path, load_config
from path_tracking.controllers import make_controller
from path_tracking.path import Path as RefPath, make_circle
from path_tracking.simulate import simulate
from path_tracking.velocity_planner import plan_speed_profile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FAIL_LAT_RMS = 0.3
SPEEDS = {"20": 5.5556, "40": 11.1111, "60": 16.6667}
SIGMAS = [0.0, 0.05, 0.10]
RADII = [15.0, 30.0, 60.0]
CTRLS = ["pure_pursuit", "stanley", "mpc"]


def run_fixed(cfg, path, speed, sigma, ctrl_key):
    veh_cfg = cfg["vehicle"]
    controller, label = make_controller(ctrl_key, cfg["controllers"], veh_cfg)
    out = simulate(
        controller, path, veh_cfg, speed,
        init={"x": path.x[0], "y": path.y[0], "yaw": path.yaw[0], "v": speed},
        settle_s=2.0, plant="kinematic", meas_noise_std=sigma, noise_seed=0,
    )
    m = out["metrics"]
    m.update({"mode": "fixed", "ctrl": ctrl_key, "sigma": sigma,
              "radius": float(path.s[-1] / (2 * np.pi) * 4 / 3),  # 近似
              "fail": bool(m["lat_rms"] > FAIL_LAT_RMS),
              "v_target": speed})
    return m


def run_planned(cfg, path, sigma, ctrl_key):
    """用规划速度剖面驱动（逐段变速）。"""
    veh_cfg = cfg["vehicle"]
    controller, label = make_controller(ctrl_key, cfg["controllers"], veh_cfg)
    curvature = path.curvature if hasattr(path, "curvature") else np.zeros(len(path.x))
    v_profile = plan_speed_profile(curvature, sigma_m=sigma, ds=0.5, a_max=2.0,
                                   v_max=16.667, v_min=1.0)

    dt = float(veh_cfg["dt"])
    from path_tracking.vehicle import KinematicBicycle
    from path_tracking.path import nearest_index
    from path_tracking.metrics import cross_track_error, heading_error, metric_pack

    veh = KinematicBicycle(wheelbase=veh_cfg["wheelbase"],
                           max_steer=veh_cfg["max_steer"],
                           max_steer_rate=veh_cfg["max_steer_rate"], dt=dt)
    veh.reset(path.x[0], path.y[0], path.yaw[0], v_profile[0])

    max_time = (path.s[-1] / max(v_profile.mean(), 1.0)) * 1.8 + 5.0
    n_max = int(max_time / dt)
    rec_x, rec_y, rec_yaw, rec_v = [], [], [], []
    ref_idx = 0
    rng = np.random.default_rng(0)
    from path_tracking.vehicle import VehicleState

    for i in range(n_max):
        s_true = veh.state
        if sigma > 0:
            obs = VehicleState(
                s_true.x + rng.normal(0, sigma),
                s_true.y + rng.normal(0, sigma),
                s_true.yaw + rng.normal(0, sigma),
                s_true.v + rng.normal(0, sigma),
            )
        else:
            obs = s_true
        # 当前参考点对应的目标速度
        idx = nearest_index(path, obs.x, obs.y, hint=ref_idx, window=80)
        v_target = float(v_profile[min(idx, len(v_profile) - 1)])
        accel, steer, ref_idx, info = controller.compute(obs, path, v_target, ref_idx, dt)
        s = veh.step(accel, steer, dt)
        rec_x.append(s.x); rec_y.append(s.y); rec_yaw.append(s.yaw); rec_v.append(s.v)
        if ref_idx >= len(path) - 2:
            break

    n = len(rec_x)
    skip = int(2.0 / dt)
    ref_x = np.zeros(n); ref_y = np.zeros(n); ref_yaw = np.zeros(n)
    hint = 0
    for i in range(n):
        hint = nearest_index(path, rec_x[i], rec_y[i], hint=hint, window=80)
        ref_x[i] = path.x[hint]; ref_y[i] = path.y[hint]; ref_yaw[i] = path.yaw[hint]
    e_lat = cross_track_error(np.array(rec_x), np.array(rec_y), np.array(rec_yaw),
                              ref_x, ref_y, ref_yaw)
    e_yaw = heading_error(np.array(rec_yaw), ref_yaw)
    sl = slice(skip, None)
    m = metric_pack(np.zeros(n)[sl], np.zeros(n)[sl], dt, e_lat[sl], e_yaw[sl])
    m.update({"mode": "planned", "ctrl": ctrl_key, "sigma": sigma,
              "fail": bool(m["lat_rms"] > FAIL_LAT_RMS),
              "v_target": float(v_profile.mean())})
    return m


def main():
    cfg = load_config(default_config_path())
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    all_runs = []
    t0 = time.time()

    for R in RADII:
        path = make_circle(radius=R, arc_deg=270.0, center_x=0.0, center_y=R)
        for ctrl_key in CTRLS:
            for sigma in SIGMAS:
                for sk, sv in SPEEDS.items():
                    m = run_fixed(cfg, path, sv, sigma, ctrl_key)
                    m["radius"] = R; m["speed_kmh"] = int(sk)
                    all_runs.append(m)
                    print(f"[fixed/{ctrl_key}/σ{sigma:.2f}/R{R:.0f}/{sk}] "
                          f"lat_rms={m['lat_rms']:.3f} fail={m['fail']}", flush=True)

                # planned：不依赖恒速（由剖面决定）
                m = run_planned(cfg, path, sigma, ctrl_key)
                m["radius"] = R; m["speed_kmh"] = "auto"
                all_runs.append(m)
                print(f"[planned/{ctrl_key}/σ{sigma:.2f}/R{R:.0f}] "
                      f"lat_rms={m['lat_rms']:.3f} fail={m['fail']}", flush=True)

    # 汇总
    summary = {}
    for mode in ("fixed", "planned"):
        summary[mode] = {}
        for ctrl_key in CTRLS:
            cells = [r for r in all_runs if r["mode"] == mode and r["ctrl"] == ctrl_key]
            if not cells:
                continue
            n_pass = sum(1 for c in cells if not c["fail"])
            summary[mode][ctrl_key] = {
                "pass": n_pass, "total": len(cells),
                "pass_rate": round(n_pass / len(cells), 3),
                "mean_lat_rms": round(float(np.mean([c["lat_rms"] for c in cells])), 4),
            }
        cells_m = [r for r in all_runs if r["mode"] == mode]
        summary[mode]["_total"] = {
            "pass": sum(1 for c in cells_m if not c["fail"]),
            "total": len(cells_m),
        }

    out = {
        "config": {"sigmas": SIGMAS, "ctrls": CTRLS, "speeds_kmh": list(SPEEDS.keys()),
                   "radii": RADII, "fail_lat_rms": FAIL_LAT_RMS},
        "summary": summary,
        "runs": all_runs,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(RESULTS / "v13_velocity_planning.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n=== 汇总 ===")
    for mode in ("fixed", "planned"):
        s = summary[mode]
        print(f"  {mode:8s} total {s['_total']['pass']}/{s['_total']['total']}  "
              f"PP {s.get('pure_pursuit',{}).get('pass','-')}  "
              f"Stan {s.get('stanley',{}).get('pass','-')}  "
              f"MPC {s.get('mpc',{}).get('pass','-')}")
    print(f"\nruntime: {out['runtime_s']}s  →  results/v13_velocity_planning.json")

    # ---- 绘图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(CTRLS))
    width = 0.35
    fixed_p = [summary["fixed"][ck]["pass"] for ck in CTRLS]
    planned_p = [summary["planned"][ck]["pass"] for ck in CTRLS]
    ax.bar(x - width/2, fixed_p, width, label="fixed（恒速）", color="#4C72B0")
    ax.bar(x + width/2, planned_p, width, label="planned（速度规划）", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(CTRLS)
    ax.set_ylabel("通过格数"); ax.set_title("E28 速度规划：失效包络通过率")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig29_velocity_planning.png", dpi=130)
    plt.close(fig)
    print(f"图 → figures/fig29_velocity_planning.png")


if __name__ == "__main__":
    main()

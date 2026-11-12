"""06 非线性轮胎饱和区（v1.3，E27）。

回应报告 §4.5："非线性轮胎（饱和区）"——当前 DynamicBicycle 线性侧偏刚度
在高速/急弯下高估轮胎力 → 控制器基于线性模型设计但真实植物饱和 → 失配。

对比：
  linear   —— 线性轮胎植物（v1.1 DynamicBicycle）
  nonlinear —— Pacejka-lite 饱和轮胎植物（μ=0.85/1.0/1.2 三档）

指标：lat_rms、失效标志、横向加速度峰值（a_y = v·r + v̇_y）、与线性基准的偏差。
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
from path_tracking.path import Path as RefPath, make_circle, make_double_lane_change
from path_tracking.simulate import simulate
from path_tracking.vehicle import DynamicBicycle, NonlinearDynamicBicycle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FAIL_LAT_RMS = 0.3
SPEEDS = {"20": 5.5556, "40": 11.1111, "60": 16.6667}
RADII = [15.0, 30.0, 60.0]
MUS = [0.85, 1.0, 1.2]
CTRLS = ["pure_pursuit", "stanley", "mpc"]


def make_dyn_plant(kind: str, mu: float, dyn_cfg: dict, dt: float):
    kwargs = dict(
        wheelbase=2.9, max_steer=0.7854, max_steer_rate=0.5236, dt=dt,
        mass=float(dyn_cfg.get("mass", 1573.0)),
        iz=float(dyn_cfg.get("iz", 2500.0)),
        lf=float(dyn_cfg.get("lf", 1.2)),
        lr=float(dyn_cfg.get("lr", 1.7)),
        cf=float(dyn_cfg.get("cf", 80000.0)),
        cr=float(dyn_cfg.get("cr", 80000.0)),
        wind_accel=0.0,
    )
    if kind == "nonlinear":
        return NonlinearDynamicBicycle(mu=mu, **kwargs)
    return DynamicBicycle(**kwargs)


def run_one(cfg, path, tag, speed, ctrl_key, plant_kind, mu=1.0):
    veh_cfg = cfg["vehicle"]
    dyn_cfg = cfg.get("dynamic_vehicle", {})
    controller, label = make_controller(ctrl_key, cfg["controllers"], veh_cfg)

    # 自定义 simulate 调用：手动管 plant（simulate 只支持 linear dynamic）
    dt = float(veh_cfg["dt"])
    veh = make_dyn_plant(plant_kind, mu, dyn_cfg, dt)
    veh.reset(path.x[0], path.y[0], path.yaw[0], speed)

    max_time = (path.s[-1] / max(speed, 1.0)) * 1.6 + 5.0
    n_max = int(max_time / dt)
    rec_x, rec_y, rec_yaw, rec_v = [], [], [], []
    rec_ay = []
    ref_idx = 0

    for i in range(n_max):
        s = veh.state
        accel, steer, ref_idx, info = controller.compute(s, path, speed, ref_idx, dt)
        s_new = veh.step(accel, steer, dt)
        # 横向加速度近似 a_y ≈ v·r（主导项）
        ay = s_new.v * veh.r
        rec_x.append(s_new.x); rec_y.append(s_new.y)
        rec_yaw.append(s_new.yaw); rec_v.append(s_new.v)
        rec_ay.append(ay)
        if ref_idx >= len(path) - 2:
            break

    # 指标（复用 metric_pack 口径）
    n = len(rec_x)
    skip = int(2.0 / dt)
    from path_tracking.metrics import cross_track_error, heading_error, metric_pack
    ref_x = np.zeros(n); ref_y = np.zeros(n); ref_yaw = np.zeros(n)
    hint = 0
    from path_tracking.path import nearest_index
    for i in range(n):
        hint = nearest_index(path, rec_x[i], rec_y[i], hint=hint, window=80)
        ref_x[i] = path.x[hint]; ref_y[i] = path.y[hint]; ref_yaw[i] = path.yaw[hint]
    e_lat = cross_track_error(np.array(rec_x), np.array(rec_y), np.array(rec_yaw),
                              ref_x, ref_y, ref_yaw)
    e_yaw = heading_error(np.array(rec_yaw), ref_yaw)
    sl = slice(skip, None)
    m = metric_pack(np.zeros(n)[sl], np.zeros(n)[sl], dt, e_lat[sl], e_yaw[sl])
    m.update({
        "scenario": tag, "plant": plant_kind, "mu": mu if plant_kind == "nonlinear" else None,
        "ctrl": ctrl_key, "speed_kmh": round(speed * 3.6),
        "radius": float(tag.replace("R", "")),
        "ay_peak": float(np.max(np.abs(np.array(rec_ay)[sl]))) if n > skip else 0.0,
        "ay_rms": float(np.sqrt(np.mean(np.array(rec_ay)[sl] ** 2))) if n > skip else 0.0,
        "fail": bool(m["lat_rms"] > FAIL_LAT_RMS),
    })
    return m


def main():
    cfg = load_config(default_config_path())
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    all_runs = []
    t0 = time.time()

    # 环形场景（曲率恒定，最容易触发饱和）
    for R in RADII:
        path = make_circle(radius=R, arc_deg=270.0, center_x=0.0, center_y=R)
        tag = f"R{R:.0f}"
        for ctrl_key in CTRLS:
            for sk, sv in SPEEDS.items():
                # 线性基准
                m = run_one(cfg, path, tag, sv, ctrl_key, "linear")
                all_runs.append(m)
                print(f"[{tag}/{ctrl_key}/{sk}kmh/linear] lat_rms={m['lat_rms']:.3f} "
                      f"fail={m['fail']} ay_pk={m['ay_peak']:.2f}", flush=True)
                # 非线性三档 μ
                for mu in MUS:
                    m = run_one(cfg, path, tag, sv, ctrl_key, "nonlinear", mu)
                    all_runs.append(m)
                    print(f"[{tag}/{ctrl_key}/{sk}kmh/NL μ={mu}] lat_rms={m['lat_rms']:.3f} "
                          f"fail={m['fail']} ay_pk={m['ay_peak']:.2f}", flush=True)

    # 汇总
    summary = {}
    for kind in ("linear", "nonlinear"):
        summary[kind] = {}
        for ctrl_key in CTRLS:
            cells = [r for r in all_runs if r["plant"] == kind and r["ctrl"] == ctrl_key]
            if not cells:
                continue
            n_pass = sum(1 for c in cells if not c["fail"])
            summary[kind][ctrl_key] = {
                "pass": n_pass, "total": len(cells),
                "mean_lat_rms": round(float(np.mean([c["lat_rms"] for c in cells])), 4),
                "mean_ay_peak": round(float(np.mean([c["ay_peak"] for c in cells])), 3),
            }

    # 非线性 vs 线性的 lat_rms 比值
    deltas = {}
    for ctrl_key in CTRLS:
        for R in RADII:
            for sk in SPEEDS:
                lin = [r for r in all_runs if r["plant"] == "linear" and r["ctrl"] == ctrl_key
                       and r.get("radius", 0) == R and r["speed_kmh"] == int(sk)]
                nl = [r for r in all_runs if r["plant"] == "nonlinear" and r["ctrl"] == ctrl_key
                      and r.get("radius", 0) == R and r["speed_kmh"] == int(sk) and r["mu"] == 1.0]
                if lin and nl:
                    key = f"{ctrl_key}/R{R:.0f}/{sk}kmh"
                    deltas[key] = {
                        "lat_rms_linear": round(lin[0]["lat_rms"], 4),
                        "lat_rms_NL_mu1.0": round(nl[0]["lat_rms"], 4),
                        "ratio": round(nl[0]["lat_rms"] / max(lin[0]["lat_rms"], 1e-9), 3),
                    }

    out = {
        "config": {"mus": MUS, "ctrls": CTRLS, "speeds_kmh": list(SPEEDS.keys()),
                   "radii": RADII, "fail_lat_rms": FAIL_LAT_RMS},
        "summary": summary,
        "deltas_NL_vs_linear": deltas,
        "runs": all_runs,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(RESULTS / "v13_nonlinear_tire.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n=== 汇总 ===")
    for kind in ("linear", "nonlinear"):
        for ck in CTRLS:
            s = summary[kind].get(ck, {})
            if s:
                print(f"  {kind:11s} {ck:13s} pass={s['pass']}/{s['total']}  "
                      f"lat_rms={s['mean_lat_rms']:.3f}  ay_pk={s['mean_ay_peak']:.2f}")
    print(f"\nruntime: {out['runtime_s']}s  →  results/v13_nonlinear_tire.json")

    # ---- 绘图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # 左：lat_rms 线性 vs NL（μ=1.0）
    labels = []
    lin_vals, nl_vals = [], []
    for ctrl_key in CTRLS:
        for R in RADII:
            for sk in SPEEDS:
                key = f"{ctrl_key}/R{R:.0f}/{sk}kmh"
                if key in deltas:
                    labels.append(key.split("/")[-1] + f"\n{ctrl_key[:2]}/R{R:.0f}")
                    lin_vals.append(deltas[key]["lat_rms_linear"])
                    nl_vals.append(deltas[key]["lat_rms_NL_mu1.0"])
    x = np.arange(len(labels))
    axes[0].bar(x - 0.15, lin_vals, 0.3, label="linear", color="#4C72B0")
    axes[0].bar(x + 0.15, nl_vals, 0.3, label="NL μ=1.0", color="#C44E52")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=5, rotation=90)
    axes[0].set_ylabel("lat_rms [m]"); axes[0].set_title("线性 vs 非线性轮胎 lat_rms")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    # 右：μ 扫描下的 lat_rms
    for ctrl_key in CTRLS:
        mu_vals = [0.85, 1.0, 1.2]
        mean_lr = []
        for mu in mu_vals:
            cells = [r for r in all_runs if r["plant"] == "nonlinear" and r["ctrl"] == ctrl_key
                     and r.get("mu") == mu]
            mean_lr.append(np.mean([c["lat_rms"] for c in cells]) if cells else 0)
        axes[1].plot(mu_vals, mean_lr, "o-", label=ctrl_key)
    axes[1].set_xlabel("μ（摩擦系数）"); axes[1].set_ylabel("mean lat_rms [m]")
    axes[1].set_title("μ 扫描下的横向误差"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig28_nonlinear_tire.png", dpi=130)
    plt.close(fig)
    print(f"图 → figures/fig28_nonlinear_tire.png")


if __name__ == "__main__":
    main()

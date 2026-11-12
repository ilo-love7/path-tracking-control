"""05 观测滤波+最近点滞回（v1.3，E26）。

回应报告 §4.5：σ≥5 cm 噪声下 PP/MPC 全灭、σ≥10 cm 全灭——"部署前必须配观测滤波/最近点滞回"。
本实验在 E9 失效包络（108 格）上加滤波+滞回，对比通过率提升。

对比条件：
  raw       —— 原始（无滤波，v1.2 E9 基线）
  filter    —— 仅 EMA 观测滤波（α=0.3）
  hyst      —— 仅最近点滞回（window=15, hyst=3）
  both      —— 滤波 + 滞回
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
from path_tracking.filtering import HysteresisNearest, StateFilter
from path_tracking.path import Path as RefPath, make_circle
from path_tracking.simulate import simulate
from path_tracking.vehicle import VehicleState

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FAIL_LAT_RMS = 0.3
SPEEDS = {"20": 5.5556, "40": 11.1111, "60": 16.6667}
SIGMAS = [0.0, 0.05, 0.10, 0.20]
RADII = [15.0, 30.0, 60.0]
CTRLS = ["pure_pursuit", "stanley", "mpc"]


class FilteredController:
    """包装器：先滤波观测，再调用内部控制器。滞回通过 ref_idx 管理。"""

    def __init__(self, inner, alpha=0.3, use_filter=True, use_hyst=True,
                 hyst_window=15, hyst_hyst=3):
        self.inner = inner
        self.filter = StateFilter(alpha=alpha) if use_filter else None
        self.hyst = HysteresisNearest(window=hyst_window, hysteresis=hyst_hyst) if use_hyst else None
        self._hyst_idx = 0

    def compute(self, obs, path, target_speed, ref_idx, dt=0.1):
        if self.filter is not None:
            z = np.array([obs.x, obs.y, obs.yaw, obs.v])
            fz = self.filter.update(z)
            obs = VehicleState(float(fz[0]), float(fz[1]), float(fz[2]), float(fz[3]))
        if self.hyst is not None:
            self.hyst._idx = ref_idx  # 同步
            new_idx = self.hyst.update(path, obs.x, obs.y)
        else:
            new_idx = ref_idx
        accel, steer, _, info = self.inner.compute(obs, path, target_speed, new_idx, dt)
        return accel, steer, new_idx, info


def run_cell(cfg, radius, speed, sigma, ctrl_key, mode):
    path = make_circle(radius=radius, arc_deg=270.0, center_x=0.0, center_y=radius)
    veh_cfg = cfg["vehicle"]
    controller, label = make_controller(ctrl_key, cfg["controllers"], veh_cfg)
    use_filter = mode in ("filter", "both")
    use_hyst = mode in ("hyst", "both")
    if mode != "raw":
        controller = FilteredController(controller, alpha=0.3,
                                        use_filter=use_filter, use_hyst=use_hyst)
    t0 = time.perf_counter()
    out = simulate(
        controller, path, veh_cfg, speed,
        init={"x": path.x[0], "y": path.y[0], "yaw": path.yaw[0], "v": speed},
        settle_s=2.0, plant="kinematic", meas_noise_std=sigma, noise_seed=0,
    )
    m = out["metrics"]
    m.update({
        "radius": radius, "speed_kmh": round(speed * 3.6), "sigma": sigma,
        "ctrl": ctrl_key, "mode": mode,
        "fail": bool(m["lat_rms"] > FAIL_LAT_RMS),
        "wall_s": time.perf_counter() - t0,
    })
    return m


def main():
    cfg = load_config(default_config_path())
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    modes = ["raw", "filter", "hyst", "both"]
    all_runs = []
    t0 = time.time()

    for mode in modes:
        for ctrl_key in CTRLS:
            for sigma in SIGMAS:
                for R in RADII:
                    for sk, sv in SPEEDS.items():
                        m = run_cell(cfg, R, sv, sigma, ctrl_key, mode)
                        all_runs.append(m)
                        tag = f"{mode}/{ctrl_key}/σ{sigma:.2f}/R{R:.0f}/{sk}kmh"
                        print(f"[{tag}] lat_rms={m['lat_rms']:.3f} fail={m['fail']}",
                              flush=True)

    # ---- 汇总：逐模式×控制器的通过率 ----
    summary = {}
    for mode in modes:
        summary[mode] = {}
        for ctrl_key in CTRLS:
            cells = [r for r in all_runs if r["mode"] == mode and r["ctrl"] == ctrl_key]
            n_pass = sum(1 for c in cells if not c["fail"])
            summary[mode][ctrl_key] = {
                "pass": n_pass, "total": len(cells),
                "pass_rate": round(n_pass / len(cells), 3),
            }
        # 模式总计
        cells_m = [r for r in all_runs if r["mode"] == mode]
        n_pass_m = sum(1 for c in cells_m if not c["fail"])
        summary[mode]["_total"] = {
            "pass": n_pass_m, "total": len(cells_m),
            "pass_rate": round(n_pass_m / len(cells_m), 3),
        }

    out = {
        "config": {"modes": modes, "ctrls": CTRLS, "sigmas": SIGMAS,
                   "radii": RADII, "speeds_kmh": list(SPEEDS.keys()),
                   "fail_lat_rms": FAIL_LAT_RMS,
                   "filter_alpha": 0.3, "hyst_window": 15, "hyst_hyst": 3},
        "summary": summary,
        "runs": all_runs,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(RESULTS / "v13_filtering_hysteresis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n=== 汇总（通过/总数） ===")
    for mode in modes:
        s = summary[mode]
        print(f"  {mode:8s} total {s['_total']['pass']}/{s['_total']['total']}  "
              f"PP {s['pure_pursuit']['pass']}  Stan {s['stanley']['pass']}  MPC {s['mpc']['pass']}")
    print(f"\nruntime: {out['runtime_s']}s  →  results/v13_filtering_hysteresis.json")

    # ---- 绘图：通过率对比条形 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(modes))
    width = 0.2
    for i, (ck, color) in enumerate(zip(CTRLS, ["#4C72B0", "#55A868", "#C44E52"])):
        vals = [summary[m][ck]["pass"] for m in modes]
        ax.bar(x + i * width, vals, width, label=ck, color=color)
    ax.set_xticks(x + width)
    ax.set_xticklabels(modes)
    ax.set_ylabel("通过格数（/36 per ctrl）")
    ax.set_title("E26 观测滤波+滞回：失效包络通过率对比")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig27_filtering_passrate.png", dpi=130)
    plt.close(fig)
    print(f"图 → figures/fig27_filtering_passrate.png")


if __name__ == "__main__":
    main()

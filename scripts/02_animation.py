"""02 动画：代表性场景 GIF（双移线-高速 / 环形-低速 × 三控制器）。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from path_tracking import plots
from path_tracking.config import default_config_path, load_config
from path_tracking.controllers import make_controller
from path_tracking.path import build_path
from path_tracking.simulate import simulate

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def main():
    cfg = load_config(default_config_path())
    veh_cfg = cfg["vehicle"]
    ctrl_cfg = cfg["controllers"]
    FIGURES.mkdir(exist_ok=True)

    cases = [
        ("double_lane_change", "high", 60.0),
        ("circle", "low", 20.0),
    ]
    for path_name, speed_name, kmh in cases:
        path = build_path(path_name, cfg["paths"][path_name])
        speed = float(cfg["speed_levels"][speed_name])
        for ctrl_key in ("pure_pursuit", "stanley", "mpc"):
            controller, label = make_controller(ctrl_key, ctrl_cfg, veh_cfg)
            out = simulate(controller, path, veh_cfg, speed,
                           init={"x": path.x[0], "y": path.y[0], "yaw": path.yaw[0], "v": speed},
                           settle_s=float(cfg["sim"]["settle_s"]))
            gif = FIGURES / f"anim_{path_name}_{speed_name}_{label}.gif"
            plots.make_animation(path, out["frame"], f"{label} · {path_name} · {kmh:.0f}km/h",
                                 gif, fps=16, stride=3)
            print(f"OK {gif.name}")


if __name__ == "__main__":
    main()

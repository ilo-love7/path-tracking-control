"""00 环境校验：casadi / 配置 / 车辆模型 smoke。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from path_tracking.config import default_config_path, load_config
from path_tracking.vehicle import KinematicBicycle


def main():
    print("== N4 环境校验 ==")
    import casadi as ca
    print(f"casadi {ca.__version__}")
    import numpy as np
    print(f"numpy {np.__version__}")

    cfg = load_config(default_config_path())
    print(f"配置键: {sorted(cfg.keys())}")
    veh = KinematicBicycle(
        wheelbase=cfg["vehicle"]["wheelbase"],
        max_steer=cfg["vehicle"]["max_steer"],
        max_steer_rate=cfg["vehicle"]["max_steer_rate"],
        dt=cfg["vehicle"]["dt"],
    )
    veh.reset(0, 0, 0, 10.0)
    for _ in range(10):
        veh.step(0.5, 0.1)
    s = veh.state
    print(f"smoke: x={s.x:.2f} y={s.y:.2f} yaw={s.yaw:.3f} v={s.v:.2f}")
    assert s.x > 0 and abs(s.yaw) > 0, "车辆模型未按预期运动"
    print("ALL_OK")


if __name__ == "__main__":
    main()

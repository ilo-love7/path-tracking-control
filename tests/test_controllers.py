from __future__ import annotations

import numpy as np
import pytest

from path_tracking.controllers import MPCController, PurePursuit, Stanley, make_controller
from path_tracking.simulate import simulate
from path_tracking.vehicle import VehicleState


def test_pure_pursuit_straight(simple_path, vehicle_cfg):
    pp = PurePursuit()
    s = VehicleState(x=0.0, y=0.0, yaw=0.0, v=5.0)
    accel, steer, _, info = pp.compute(s, simple_path, 5.0, 0)
    assert abs(steer) < 0.05  # 直线上近似直行
    assert accel == pytest.approx(0.0, abs=1e-6)


def test_stanley_straight(simple_path, vehicle_cfg):
    st = Stanley()
    s = VehicleState(x=0.0, y=0.0, yaw=0.0, v=5.0)
    accel, steer, _, info = st.compute(s, simple_path, 5.0, 0)
    assert abs(steer) < 0.1


def test_stanley_corrects_left_offset(simple_path):
    st = Stanley()
    s = VehicleState(x=0.0, y=0.5, yaw=0.0, v=5.0)  # 偏左
    _, steer, _, info = st.compute(s, simple_path, 5.0, 0)
    assert steer < 0  # 应向右打


def test_mpc_instantiation(vehicle_cfg):
    mpc = MPCController(horizon=3)
    assert mpc._solver is not None


def test_closed_loop_straight_pp(simple_path, vehicle_cfg):
    pp = PurePursuit()
    out = simulate(pp, simple_path, vehicle_cfg, target_speed=5.0,
                   init={"x": 0, "y": 0, "yaw": 0, "v": 5.0}, settle_s=0.5)
    m = out["metrics"]
    assert m["lat_rms"] < 0.1
    assert m["n_steps"] > 10


def test_closed_loop_circle_stanley(circle_path, vehicle_cfg):
    st = Stanley()
    out = simulate(st, circle_path, vehicle_cfg, target_speed=5.0,
                   init={"x": circle_path.x[0], "y": circle_path.y[0],
                         "yaw": circle_path.yaw[0], "v": 5.0}, settle_s=1.0)
    m = out["metrics"]
    assert m["lat_rms"] < 2.0  # Stanley 在圆弧上允许一定稳态误差


def test_make_controller_factory(vehicle_cfg):
    ctrl_cfg = {
        "pure_pursuit": {"k": 0.1, "lfc": 2.0},
        "stanley": {"k": 0.5, "k_soft": 1.0},
        "mpc": {"horizon": 3, "q": [1, 1, 0.5, 0.5], "r": [0.01, 0.01], "rd": [0.01, 1.0],
                "max_accel": 3.0, "max_decel": -5.0},
    }
    for name, label in [("pure_pursuit", "PurePursuit"), ("stanley", "Stanley"), ("mpc", "MPC")]:
        c, lab = make_controller(name, ctrl_cfg, vehicle_cfg)
        assert lab == label

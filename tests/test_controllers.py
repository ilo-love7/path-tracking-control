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
    mpc._build_solver(3)
    assert 3 in mpc._solvers


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


# ---------------- v1.1 自适应 horizon / 曲率前馈 ----------------

def test_mpc_adaptive_horizon_selection(vehicle_cfg):
    from path_tracking.controllers import MPCController
    ctrl = MPCController(horizon_mode="adaptive", preview_m=30.0,
                         horizon_min=5, horizon_max=20, dt=0.1)
    # N = ceil(30/(v*0.1)) clip [5,20]：低速 → 上限 20；v=60 → 5（下限）
    assert ctrl._select_horizon(2.0) == 20
    assert ctrl._select_horizon(60.0) == 5
    assert ctrl._select_horizon(15.0) == 20   # 30/1.5=20
    assert ctrl._select_horizon(100.0) == 5   # 30/10=3 → clip 5


def test_mpc_adaptive_solver_cache(vehicle_cfg):
    from path_tracking.controllers import MPCController
    ctrl = MPCController(horizon_mode="adaptive", preview_m=20.0,
                         horizon_min=4, horizon_max=12, dt=0.1)
    n1 = ctrl._select_horizon(15.0)   # 20/1.5≈14 → 12
    ctrl._build_solver(n1)
    assert n1 in ctrl._solvers
    assert 99 not in ctrl._solvers


def test_mpc_rollout_dims(vehicle_cfg):
    import numpy as np
    from path_tracking.controllers import MPCController
    ctrl = MPCController(horizon=4)
    x0 = np.array([0.0, 0.0, 5.0, 0.0])
    u = np.zeros((2, 4))
    X = ctrl._rollout(x0, u, 4)
    assert X.shape == (4, 5)
    # 直行无转角：x 前进，y≈0
    assert X[0, -1] > 1.0
    assert abs(X[1, -1]) < 1e-6


def test_stanley_ff_reduces_circle_error(circle_path, vehicle_cfg):
    from path_tracking.controllers import Stanley
    from path_tracking.simulate import simulate
    init = {"x": circle_path.x[0], "y": circle_path.y[0],
            "yaw": circle_path.yaw[0], "v": 5.0}
    st_plain = Stanley()
    st_ff = Stanley(curvature_ff=True)
    out_p = simulate(st_plain, circle_path, vehicle_cfg, 5.0, init=init, settle_s=1.0)
    out_f = simulate(st_ff, circle_path, vehicle_cfg, 5.0, init=init, settle_s=1.0)
    # 曲率前馈应显著降低恒曲率稳态误差
    assert out_f["metrics"]["lat_rms"] < out_p["metrics"]["lat_rms"] * 0.6


def test_make_controller_new_names(vehicle_cfg):
    ctrl_cfg = {
        "pure_pursuit": {"k": 0.1, "lfc": 2.0},
        "stanley": {"k": 0.5, "k_soft": 1.0},
        "mpc": {"horizon": 3, "q": [1, 1, 0.5, 0.5], "r": [0.01, 0.01], "rd": [0.01, 1.0],
                "max_accel": 3.0, "max_decel": -5.0, "preview_m": 30.0,
                "horizon_min": 5, "horizon_max": 15},
    }
    c1, l1 = make_controller("stanley_ff", ctrl_cfg, vehicle_cfg)
    assert l1 == "Stanley+FF" and c1.curvature_ff
    c2, l2 = make_controller("mpc_adaptive", ctrl_cfg, vehicle_cfg)
    assert l2 == "MPC-A" and c2.horizon_mode == "adaptive"


def test_mpc_term_w_passthrough(vehicle_cfg):
    """term_w 可配置：工厂透传，默认 5.0，可设 0（消融实验用）。"""
    from path_tracking.controllers import MPCController, make_controller
    ctrl_cfg = {
        "mpc": {"horizon": 3, "q": [1, 1, 0.5, 0.5], "r": [0.01, 0.01], "rd": [0.01, 1.0],
                "max_accel": 3.0, "max_decel": -5.0, "term_w": 0.0},
    }
    c1, _ = make_controller("mpc", ctrl_cfg, vehicle_cfg)
    assert c1.term_w == 0.0
    ctrl_cfg["mpc"].pop("term_w")
    c2, _ = make_controller("mpc", ctrl_cfg, vehicle_cfg)
    assert c2.term_w == 5.0
    assert MPCController().term_w == 5.0

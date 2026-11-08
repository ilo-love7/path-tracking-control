from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# casadi 插件 DLL 依赖需在 PATH 上（必须在 import casadi 之前设置）。
# 不要用 os.add_dll_directory：会切到 LOAD_LIBRARY_SEARCH_* 语义反而丢同目录依赖。
_casadi_dir = Path(sys.prefix) / "Lib" / "site-packages" / "casadi"
if _casadi_dir.exists():
    _p = str(_casadi_dir)
    if _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")


@pytest.fixture
def simple_path():
    """直线 100 m 路径。"""
    from path_tracking.path import Path
    s = np.arange(0, 100, 0.5)
    x = s.copy()
    y = np.zeros_like(s)
    yaw = np.zeros_like(s)
    curv = np.zeros_like(s)
    return Path(x, y, yaw, curv, s)


@pytest.fixture
def circle_path():
    from path_tracking.path import make_circle
    return make_circle(radius=30.0, arc_deg=270.0, center_x=0.0, center_y=30.0, ds=0.5)


@pytest.fixture
def dlc_path():
    from path_tracking.path import make_double_lane_change
    pts = [
        {"x": 0.0, "y": 0.0}, {"x": 35.0, "y": 0.0}, {"x": 50.0, "y": 3.5},
        {"x": 75.0, "y": 3.5}, {"x": 90.0, "y": 0.0}, {"x": 130.0, "y": 0.0},
    ]
    return make_double_lane_change(pts)


@pytest.fixture
def vehicle_cfg():
    return {"wheelbase": 2.9, "max_steer": 0.7854, "max_steer_rate": 0.5236, "dt": 0.1}

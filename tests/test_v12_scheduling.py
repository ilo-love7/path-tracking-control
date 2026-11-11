"""v1.2 单测：Stanley 增益调度纯函数。"""
import numpy as np
import pytest

from path_tracking.scheduling import stanley_gain_schedule


def test_const_mode():
    assert stanley_gain_schedule(3.0, 0.5, "const") == 0.5
    assert stanley_gain_schedule(30.0, 0.5, "const") == 0.5


def test_linear_anchor_and_monotonic():
    v_ref = 5.5556
    assert abs(stanley_gain_schedule(v_ref, 0.5, "linear", v_ref=v_ref, alpha=0.05) - 0.5) < 1e-9
    vals = [stanley_gain_schedule(v, 0.5, "linear", v_ref=v_ref, alpha=0.05)
            for v in (2.0, 5.0, 10.0, 20.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_speed_inv_invariant_feedback():
    # 反馈项 k_eff·e/(v+k_soft) 应速度不变
    v_ref, k_soft, k0 = 5.5556, 1.0, 0.5
    e = 0.3
    terms = []
    for v in (1.0, 5.0, 15.0, 30.0):
        k_eff = stanley_gain_schedule(v, k0, "speed_inv", v_ref=v_ref, k_soft=k_soft)
        terms.append(k_eff * e / (v + k_soft))
    assert max(terms) - min(terms) < 1e-9


def test_speed_inv_anchor():
    v_ref = 5.5556
    assert abs(stanley_gain_schedule(v_ref, 0.5, "speed_inv", v_ref=v_ref) - 0.5) < 1e-9


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        stanley_gain_schedule(5.0, 0.5, "nope")


def test_negative_speed_clipped():
    assert stanley_gain_schedule(-3.0, 0.5, "linear", alpha=0.1) == \
        stanley_gain_schedule(0.0, 0.5, "linear", alpha=0.1)


def test_stanley_uses_schedule():
    from path_tracking.controllers import Stanley
    st = Stanley(k=0.5, k_schedule="speed_inv")
    assert st.k_schedule == "speed_inv"
    # 高速下 speed_inv 的有效增益大于 const
    assert stanley_gain_schedule(20.0, st.k, "speed_inv") > st.k

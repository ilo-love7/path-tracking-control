# 车辆路径跟踪控制器仿真对比（N4）

运动学自行车模型（Kong et al., IEEE IV 2015）下，**Pure Pursuit / Stanley / 非线性 MPC** 三类横向路径跟踪控制器统一对比：双移线 + 环形路径 × 20/60 km/h 工况，跟踪精度、控制平滑度、计算耗时全口径评测。

> 博远线上科研提案 N4 ｜ 汽车/控制 ｜ 数值仿真 ｜ v1.0.0 ｜ 2026-11-09

## 结果摘要

| 场景 | 指标 | PurePursuit | Stanley | **MPC** |
|---|---|---|---|---|
| 双移线 20 km/h | lat_rms [m] | 0.0077 | 0.0408 | **0.0040** |
| 双移线 60 km/h | lat_rms [m] | **0.0081** | 0.0239 | 0.1264 |
| 环形 20 km/h | lat_rms [m] | 0.0222 | 0.1319 | **0.0113** |
| 环形 60 km/h | lat_rms [m] | **0.0203** | 0.1027 | 0.1431 |

1. **低速 MPC 最优**（lat_rms 0.004/0.011 m）：预测视野覆盖路径变化，提前规划；
2. **高速 PurePursuit 最优**（0.008/0.020 m）：N=5×dt=0.1 s 的 MPC 视野仅 8 m，双移线处滞后（lat_max 0.75 m）且转角速率 RMS 高 15 倍；
3. **Stanley 最平滑**（低速 steer_rate 0.028 rad/s）但曲线上有经典稳态误差（环形 lat_rms 0.13 m）；
4. MPC 求解 5–6 ms/步（IPOPT，horizon=5），10 Hz 控制可行；
5. 三类控制器接口统一（输出前轮转角），指标口径一致，可复现。

## 运行

```bash
pip install -r requirements.txt
bash run_all.sh        # 00 校验 → 01 对比实验 → 02 动画
python -m pytest tests/ -q
```

PythonRobotics 参考实现（参数对齐基准）：`pure_pursuit.py / stanley_controller.py / model_predictive_speed_and_steer_control.py`（AtsushiSakai/PythonRobotics，7 万+ star）。

## 结构

```
configs/config.yaml    车辆参数/路径定义/控制器超参/速度档
src/path_tracking/     vehicle(自行车模型) / path(样条路径) / metrics / controllers(PP/Stanley/MPC) / simulate / plots
scripts/00-02          校验 → 对比实验 → 动画
tests/                 24 项单测
reports/               基准评测报告-N4.md / 工作记录-N4.md
results/               comparison_metrics.json / comparison_rows.csv
figures/               15 张 PNG + 6 个 GIF 动画
```

## 结论

1. **没有免费午餐**：MPC 低速领先、几何法高速领先，与预测视野长度相对路径变化尺度的比值有关；
2. PurePursuit 自适应前瞻（Ld = 2.0 + 0.1·v）天然随速度伸长，是高速下稳健的原因；
3. Stanley 结构最简、最平滑，但恒曲率路径稳态误差大（k=0.5 时 0.13 m）；
4. MPC 转角平滑度差（steer_jerk RMS 高 1–2 个量级）源于短视野内追赶参考的急促修正，可用速度自适应 horizon/Rd 改善。

## 未做项（如实标注）

- 动力学自行车模型（轮胎侧偏/滑移，Kong 2015 动力学版）
- 视野长度自适应 MPC（速度调度 horizon / 终端代价）
- 变曲率/避障路径、闭环速度-转向耦合控制
- 与 PythonRobotics 逐位数值对齐验证（参数量级已对齐，实现为独立代码）

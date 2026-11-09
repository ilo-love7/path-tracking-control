# 车辆路径跟踪控制器仿真对比（N4）

运动学自行车模型（Kong et al., IEEE IV 2015）下，**Pure Pursuit / Stanley / 非线性 MPC** 三类横向路径跟踪控制器统一对比：双移线 + 环形 + clothoid 变曲率路径 × 20/60 km/h 工况，跟踪精度、控制平滑度、计算耗时、鲁棒性全口径评测。

> 博远线上科研提案 N4 ｜ 汽车/控制 ｜ 数值仿真 ｜ v1.1.0 ｜ 2026-11-09

## 结果摘要（v1.0 基准）

| 场景 | 指标 | PurePursuit | Stanley | **MPC** |
|---|---|---|---|---|
| 双移线 20 km/h | lat_rms [m] | 0.0077 | 0.0408 | **0.0040** |
| 双移线 60 km/h | lat_rms [m] | **0.0081** | 0.0239 | 0.1264 |
| 环形 20 km/h | lat_rms [m] | 0.0222 | 0.1319 | **0.0113** |
| 环形 60 km/h | lat_rms [m] | **0.0203** | 0.1027 | 0.1431 |

1. **低速 MPC 最优**（lat_rms 0.004/0.011 m）：预测视野覆盖路径变化，提前规划；
2. **高速 PurePursuit 最优**（0.008/0.020 m）：MPC 反而退化（lat_max 0.75 m、转角速率 RMS 高 15 倍）——「速度反转」现象；
3. **Stanley 最平滑**（低速 steer_rate 0.028 rad/s）但恒曲率有经典稳态误差（0.13 m）；
4. MPC 求解 5–6 ms/步（IPOPT，N=5），10 Hz 控制可行。

## v1.1 深化发现（36 项单测，78 行指标）

5. **速度反转根因定论**：加长 horizon 假说被证伪（N≥10 误差放大 12–43 倍、IPOPT 局部解混乱）；终端代价有害（去掉后高速误差降 4.8 倍）但不足以翻盘；本质是短窗参考采样 + PP 误差反馈的渐近稳定结构优势。
6. **Stanley+曲率前馈**（后轴误差口径）：恒曲率稳态误差 33 倍改善（环形低速 0.132→0.004 m），反超 MPC。
7. **鲁棒性分层**：侧风（0.5 m/s²）下 MPC 最优；量测噪声（σ=5 cm）下 Stanley 最稳，PP/MPC 因最近点索引跳动崩坏（12–24 m）。
8. **模型失配警戒线**：2-DOF 动力学植物 + 高速环形，运动学 MPC 完全发散（82 m）、PP 发散（4.4 m），Stanley 1.5 m 最稳——优化型控制器放大失配，简单反馈保命。
9. 自适应 horizon（MPC-A，预瞄距离恒定）全面差于固定 N=5，负结果如实报告。

## 运行

```bash
pip install -r requirements.txt
bash run_all.sh        # 00 校验 → 01 对比 → 03 深化 → 02 动画
python -m pytest tests/ -q   # 36 项
```

PythonRobotics 参考实现（参数对齐基准）：`pure_pursuit.py / stanley_controller.py / model_predictive_speed_and_steer_control.py`（AtsushiSakai/PythonRobotics，7 万+ star）。

## 结构

```
configs/config.yaml    车辆参数/动力学参数/路径/控制器超参/速度档
src/path_tracking/     vehicle(运动学+动力学自行车) / path(样条+clothoid) / metrics
                       controllers(PP/Stanley/Stanley+FF/MPC/MPC-A) / simulate(扰动注入) / plots
scripts/00-03          校验 → 对比实验 → 深化实验 → 动画
tests/                 36 项单测
reports/               基准评测报告-N4.md（§3 深化实验）/ 工作记录-N4.md
results/               comparison_metrics.json + deepening_metrics.json（78 行）
figures/               轨迹/误差/消融图 + GIF 动画
```

## 结论

1. **没有免费午餐**：MPC 低速领先、几何法高速领先；且优势区间受终端代价、模型匹配度、噪声水平共同调制；
2. PurePursuit 自适应前瞻（Ld = 2.0 + 0.1·v）随速度伸长，是高速稳健的结构性原因；
3. Stanley+FF 是曲线路况的性价比之王（33 倍稳态改善，<0.1 ms）；
4. 噪声/模型失配场景下简单反馈结构反超优化型控制器——部署前先认清误差源。

## 未做项（如实标注）

- 非线性轮胎模型（饱和区/滑移）
- MPC 内嵌动力学模型（修复 E1 失配）
- 观测滤波/最近点滞回（E5 噪声崩坏的工程解）
- 与 PythonRobotics 逐位数值对齐验证（参数量级已对齐，实现为独立代码）

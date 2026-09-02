# NetCatch NUC 弹道预测操作手册

三台 Jetson 的 UDP 通信已经验证后，当前阶段从本文件继续。predictor 只在 NUC 上运行，订阅 `obj1` 和 payload `p11` 的动捕数据，并发布 `/netcatch/dynamics/prediction`。

## 1. 哪些操作需要重复执行

| 操作 | 在哪里执行 | 执行频率 |
|---|---|---|
| 同步代码、安装依赖、修改 `config.json` | Mac 或 NUC | 第一次部署、代码或参数改变时 |
| 启动 predictor | NUC | 每次实验启动一次 |
| 检查三个 VRPN topic | NUC | 每次实验开始前建议检查一次 |
| 调用 `arm` | NUC | 每次投掷前一次，不需要与离手同步 |
| 调用 `rearm` | NUC | 上一次投掷进入 `DONE` 后、下一次投掷前 |
| 调用 `cancel` | NUC | 仅需要提前终止本次投掷时 |

Mac 和 Jetson 都不运行 predictor。

## 2. 已固定的配置

文件为 `~/20260829/nuc/predict/config.json`，当前关键值为：

- 物体刚体：`obj1`
- payload 刚体：`p11`
- 初始 payload target：`[-0.6, 0.6, 0.4] m`
- payload 实际飞行高度：`0.4 m`
- 预测截面高度：`0.833 m`
- 输出频率：`30 Hz`
- 单次有效预测窗口：最多 `5 s`
- release 模式：`armed_auto`
- 自动检测缓存：`0.25 s`
- 自由飞行拟合窗口：`0.05 s`，pose/twist 各至少 `8` 个样本
- 候选最小速度：`0.35 m/s`
- 最大位置/速度模型残差：`0.012 m` / `0.20 m/s`
- 最大拟合加速度误差：`4.0 m/s²`（相对 `[0, 0, -9.81] m/s²`）
- 连续确认窗口：`2`

没有修改算法参数时，不需要每次开机编辑该文件。

## 2.1 CSV 调试日志（可选）

需要记录每次投掷的调试明细时，用 `--csv` 指定一个基准路径，或设置环境变量 `NETCATCH_PREDICT_CSV`（两者都不设置则关闭）：

```bash
cd ~/20260829/nuc/predict
./run_predict.sh --csv logs/predict.csv
# 等价：NETCATCH_PREDICT_CSV=logs/predict.csv ./run_predict.sh
```

每次启动会生成带时间戳的新文件（如 `logs/predict_20260902_101500_123456.csv`），不覆盖历史；目录不存在会自动创建。日志只在 **EKF 启动后**（自由飞行确认/手动 release、滤波器首次接受测量起）以发布频率 30 Hz 逐行追加，每行包含：

- 时间与状态：`t_monotonic_s`、`ekf_elapsed_s`（EKF 已启动计时，秒）、`throw_id`、`state`、`valid`（有效位）、`reason`（无目标预测/不发布的原因，如 `mapping_not_committed`、`no_future_descending_crossing`、`estimator_not_ready` 等）
- 球实际位置：`pose_measured_*`/`twist_measured_*`（最近一次接收的原始测量）、`ekf_*`（EKF 估计位置与速度）
- 预测结果：`intercept_*`、`time_to_contact_s`、`intercept_time_ns`、`xy_radius_95_m`、`confidence`、`command_target_*`、`hold_target_*`
- 实际捕获点：`actual_capture_*` + `actual_capture_time_ns` —— 球 obj1 实际下降穿过预测平面（z=0.833）的位置，由平面上下两个原始 pose 样本线性插值得到，可与预测 `intercept_*` 直接对比评估预测精度；球未穿过平面时为空
- 数据时间戳：`state_time_ns`（滤波状态时刻）

进入 `DONE`/`ERROR` 后对象订阅销毁，原始测量列会清空（空值），不会再用过期数据制造假残差；`ekf_*` 与 `actual_capture_*` 保留最后一次投掷的最终结果。写 CSV 失败只会在日志中报错并禁用记录，不会影响预测进程。

## 3. 每次实验开始时在 NUC 执行

### 3.1 NUC 终端 A：启动 predictor
```bash
source /opt/ros/humble/setup.bash
```
```bash
cd ~/20260829/nuc/predict
./run_predict.sh
```

正常启动时会看到：

```text
waiting to arm automatic free-flight detection; object=obj1, frame=mocap_world_enu
```

保持该终端运行。

### 3.2 NUC 终端 B：检查动捕数据

```bash
source /opt/ros/humble/setup.bash

ros2 topic echo /vrpn/obj1/pose geometry_msgs/msg/PoseStamped --once
ros2 topic echo /vrpn/obj1/twist geometry_msgs/msg/TwistStamped --once
ros2 topic echo /vrpn/p11/pose geometry_msgs/msg/PoseStamped --once

ros2 topic hz /vrpn/obj1/pose
```

最后一条命令看到约 `200 Hz` 后按 `Ctrl-C`。如果三个 `echo --once` 都能立即收到数据，并且 `obj1` pose 约为 `200 Hz`，就可以进行投掷测试。

### 3.3 NUC 终端 C：启动 UDP sender

```bash
cd ~/20260829/nuc/udp
./run_udp_sender.sh ./config.json
```

sender 的具体说明见 [`../udp/README.md`](../udp/README.md)。

### 3.4 NUC 终端 D：低噪声观察状态

```bash
cd ~/20260829/nuc/udp
./run_udp_receiver.sh ./config.json --count 0 --timeout-s 5 --changes-only
```

这里只在状态变化时输出一行，不需要使用连续刷屏的 `--json`。

## 4. 每次投掷的操作

### 第一次投掷

1. 确认 predictor 为 `WAIT_RELEASE valid=false reason=waiting_for_arm`。
2. 物体放置完成、准备手抛时，在 **NUC 终端 B** 调用：

```bash
ros2 service call /netcatch/dynamics/arm std_srvs/srv/Trigger '{}'
```

3. 确认 receiver 显示：

```text
state=WAIT_RELEASE valid=false reason=armed_waiting_for_free_flight
```

4. 看到该状态后直接手抛。`arm` 可以提前数秒调用，不需要与物体离手同步。

`success=True` 只表示自动检测已经使能。predictor 会缓存最近 `0.25 s` 的 `obj1` pose/twist，用连续两个 `0.05 s` 窗口确认轨迹符合自由飞行；确认后自动回溯到确认窗口内的自由飞行起点、重置 EKF、回放自由飞行样本并进入 `TRACKING`。真正形成可下发预测要看：

```text
state=TRACKING valid=true
```

首次有效预测前会先出现若干条 `TRACKING valid=false`。有效消息中的 target 高度应始终为 `0.4`，intercept 截面高度为 `0.833`。

### 下一次投掷

上一次进入 `DONE` 后，先在 **NUC** 执行：

```bash
ros2 service call /netcatch/dynamics/rearm std_srvs/srv/Trigger '{}'
ros2 service call /netcatch/dynamics/arm std_srvs/srv/Trigger '{}'
```

确认依次看到 `rearmed_waiting_for_arm` 和 `armed_waiting_for_free_flight`，然后直接手抛。如果需要提前终止当前投掷：

```bash
ros2 service call /netcatch/dynamics/cancel std_srvs/srv/Trigger '{}'
```

`/netcatch/dynamics/release` 会绕过自动自由飞行检测，只保留给故障隔离；正式 demo 不执行：

```bash
ros2 service call /netcatch/dynamics/release std_srvs/srv/Trigger '{}'
```

## 5. 如何判断预测阶段是否通过

一次投掷至少应观察到：

1. 启动或 rearm 后为 `WAIT_RELEASE valid=false reason=waiting_for_arm` 或 `rearmed_waiting_for_arm`
2. arm 后仍为 `WAIT_RELEASE valid=false`，但 reason 变为 `armed_waiting_for_free_flight`
3. 持球和摆臂阶段不进入 `TRACKING`
4. 自由飞行确认后自动进入 `TRACKING`，predictor 日志出现 `free flight detected`
5. 连续稳定后出现 `TRACKING valid=true`
6. target 的 XY 是有限值且不会来回跳动，Z 始终为 `0.4`
7. 物体数据丢失时进入 `LOST valid=false`，恢复后重新积累稳定预测
8. 本次投掷结束后进入 `DONE valid=false`

如果 arm 后始终没有自动进入 `TRACKING`，或进入后直接 `LOST`，先在 NUC 重新检查：

```bash
ros2 topic hz /vrpn/obj1/pose
ros2 topic hz /vrpn/obj1/twist
```

然后查看低噪声 receiver 输出中的 `reason=`。不要先用完整 `--json` 刷屏；只有需要核对时间戳等完整字段时才临时加 `--json --count 5`。

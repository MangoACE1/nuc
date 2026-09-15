# NetCatch NUC 动态目标操作入口

`predict/` 是 `obj1` 弹道预测进程，`udp/` 是 ROS-to-UDP 组播发送与诊断进程；两者是独立生产进程，只通过 NUC 本机的 `std_msgs/msg/String` topic `/netcatch/dynamics/prediction` 通信。实现与参数细节见 [预测器说明](predict/README.md) 和 [UDP 说明](udp/README.md)。

日常实验只需要本文件第 3 节的 NUC 启动步骤。第 1 节仅在部署或代码更新时执行，第 2 节是配置及实验前检查；Mac 不运行 predictor/sender，Jetson 不运行 predictor。

## 1. 开发检查与部署

NUC 使用 Ubuntu 22.04、ROS 2 Humble，并安装 Python 3、NumPy、`rclpy`、`geometry_msgs`、`std_msgs`、`std_srvs`。VRPN bridge 必须已提供配置中的 `PoseStamped`/`TwistStamped` topics；本目录不负责启动或修改 VRPN。

部署前在 **Mac 的当前仓库终端**运行完整测试；`test/` 不复制到 NUC：

```bash
cd /home/steve/Documents/202603/NetCatch/script/dynamics
NETCATCH_PYTHON=/home/steve/miniconda3/envs/genesis/bin/python ./verify_dev.sh
```

仍在 **Mac 的当前仓库终端**只复制 `nuc/`。下面的 `--exclude 'config.json'` 用于保留 NUC 上已经填写好的现场配置；首次部署尚无 NUC 配置时去掉该选项：

```bash
DYNAMICS_SRC=/home/steve/Documents/202603/NetCatch/script/dynamics
rsync -av --exclude 'config.json' "${DYNAMICS_SRC}/nuc/" lzh@10.1.1.115:/home/lzh/20260829/nuc/
```

复制后在 **NUC** 上检查部署包本身：

```bash
cd ~/20260829/nuc
chmod +x verify_nuc.sh predict/run_predict.sh udp/run_udp_sender.sh udp/run_udp_receiver.sh
NETCATCH_PYTHON=python3 ./verify_nuc.sh
```

`verify_nuc.sh` 只检查契约、生产模块编译和 launcher 语法，不依赖 pytest 或 `test/`，也不证明 ROS/VRPN/组播链路可用。

## 2. 飞行前坐标与配置

在 NUC 和每台 Jetson 上，把 `/vrpn/p11/pose` 分别放到至少三个不共线的已知位置进行对比；确认各机数值一致、轴向一致、单位为米、时间戳持续更新。随后在 NUC 检查：

```bash
ros2 topic type /vrpn/p11/pose
ros2 topic type /vrpn/obj1/pose
ros2 topic type /vrpn/obj1/twist
ros2 topic hz /vrpn/obj1/pose
ros2 topic hz /vrpn/obj1/twist
```

类型应分别为 `geometry_msgs/msg/PoseStamped`、`geometry_msgs/msg/PoseStamped`、`geometry_msgs/msg/TwistStamped`，`obj1` pose/twist 实际交付率应约为 200 Hz。

只有第一次配置或 NUC 的 IP 改变时，才编辑 `udp/config.json` 的 `multicast_interface_ip`；填写 NUC 实际用于通信的网卡 IPv4，当前 Wi-Fi 地址为 `10.1.1.115`。不需要每次开机修改。不要改 group/port/frame/object，也不要改固定初始 payload target `[-0.6, 0.6, 0.4]`；修改后重跑 `./verify_nuc.sh`。

## 3. 精确启动顺序

### NUC 终端 A：先启动预测器

```bash
source /opt/ros/humble/setup.bash
cd ~/20260829/nuc/predict
./run_predict.sh
```

### NUC 终端 B：再启动唯一的 UDP sender

```bash
source /opt/ros/humble/setup.bash
cd ~/20260829/nuc/udp
./run_udp_sender.sh ./config.json
```

### NUC 终端 C：可选的低噪声状态观察

```bash
cd ~/20260829/nuc/udp
./run_udp_receiver.sh ./config.json --count 0 --timeout-s 5 --changes-only
```

### NUC 终端 D：每次手抛前 arm

```bash
source /opt/ros/humble/setup.bash
ros2 service call /netcatch/dynamics/arm std_srvs/srv/Trigger '{}'
```

看到 `reason=armed_waiting_for_free_flight` 后直接手抛，不需要让命令与离手同步。取消当前 throw；再次使用前先 rearm，再重新 arm：

```bash
ros2 service call /netcatch/dynamics/cancel std_srvs/srv/Trigger '{}'
ros2 service call /netcatch/dynamics/rearm std_srvs/srv/Trigger '{}'
ros2 service call /netcatch/dynamics/arm std_srvs/srv/Trigger '{}'
```

## 4. 操作状态要点

- arm 只使能自动自由飞行检测，service 的 `success=true` 不代表已经离手、预测成功或接物成功。arm 后保持 `WAIT_RELEASE/valid=false`，reason 为 `armed_waiting_for_free_flight`。
- predictor 缓存最近 `0.25 s` 的 `obj1` pose/twist，并用连续两个 `0.05 s` 弹道一致性窗口排除持球和摆臂运动；确认后回溯、清空 EKF、只回放自由飞行样本并自动进入 `TRACKING`。
- `/netcatch/dynamics/release` 仅是绕过检测的强制调试入口，正式 demo 不执行。
- 第一个有效 target 前必须连续得到 3 个稳定预测；前 2 个仍为 `TRACKING/valid=false`。第 3 个首次有效 target 才启动 5 s active budget 和预测接触 deadline。
- 动捕数据短时断流（< `0.25 s`）时保持 `TRACKING/valid=true`，沿用上一份有效预测（整包冻结）；断流超过 `0.25 s` 直接进入 `DONE`（结束本次投掷、取消对象订阅）。
- active budget 在 `LOST/RECOVERING` 暂停，预测接触 deadline 不暂停。
- `DONE` 会取消 `obj1` pose/twist 订阅，直到 rearm 后才重新订阅。

## 5. 验收边界

开发验收：开发机上的 `./verify_dev.sh` 返回 0，统一测试全部通过。部署包验收：NUC 上的 `./verify_nuc.sh` 返回 0，契约、两组生产模块编译和三个 launcher 的 `bash -n` 全部通过。

现场验收：完成三点 VRPN 坐标对比；确认 `obj1` 类型、时间戳与约 200 Hz 真正交付；预测 topic 约 30 Hz；验证 arm 后持球/摆臂不误触发、手抛后自动进入 `TRACKING`、3 次稳定门控、`LOST`/恢复、cancel/rearm；receiver 与 `tcpdump` 均看到从指定通信接口发出的 `239.255.42.99:15150` 组播，约 30 Hz、单包不超过 1200 B。**注意：预测频率统一改为60Hz,增大频率可增加收敛速度。**

当前 Wi-Fi 网络上的 NUC 到三台 Jetson 组播已经完成测试；以后更换 Wi-Fi、有线接口或网段时，需要重新检查各机 IP 和组播接收。正式 Jetson `dynamics` receiver 与 policy 接入仍是后续工作。

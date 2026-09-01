# NetCatch UDP 组播操作手册

本目录只负责把 NUC 上的 `/netcatch/dynamics/prediction` 转成 UDP 组播包。三台 Jetson 已经完成通信测试后，预测阶段请继续看 [`../predict/README.md`](../predict/README.md)，不用重新执行本文件里的部署和三机验收操作。

## 1. Mac、NUC、Jetson 分别做什么

| 设备 | 作用 | 实验运行时是否需要操作 |
|---|---|---|
| Mac（当前代码仓库所在的开发环境） | 修改代码，并把代码同步到 NUC 和 Jetson | 不运行 predictor、sender 或 receiver |
| NUC（`10.1.1.115`） | 运行 predictor 和唯一的 UDP sender | 每次实验启动 predictor 和 sender |
| 三台 Jetson | 接收同一份组播 target；调试时可运行诊断 receiver | 诊断 receiver 不必每次启动；以后由正式 `dynamics` receiver 接收 |

UDP 接收设备就是三台 Jetson。

## 2. 哪些命令需要重复执行

| 操作 | 在哪里执行 | 执行频率 |
|---|---|---|
| 安装依赖、`chmod` | NUC 或 Jetson | 第一次部署时一次 |
| `rsync` 同步代码 | Mac | 代码更新后执行，不是每次开机执行 |
| 修改 `multicast_interface_ip` | NUC 或对应 Jetson | 第一次配置或该设备 IP 改变时执行 |
| 启动 predictor | NUC | 每次实验启动一次 |
| 启动 UDP sender | NUC | 每次实验启动一次，且只能有一个 sender |
| 启动诊断 receiver、运行 `tcpdump` | NUC 或 Jetson | 仅调试时执行 |

Mac 在正常实验运行期间不需要执行任何 UDP 命令。

## 3. 仅在部署或代码更新后执行

### 3.1 Mac：同步 NUC 代码

下面的命令在 **Mac 的当前仓库终端**执行。NUC 已有现场 `config.json` 时保留 `--exclude 'config.json'`；全新部署时去掉该选项。

```bash
rsync -av --exclude 'config.json' \
  "/Users/steve/Documents/code/NetCatch/script/dynamics/nuc/" \
  lzh@10.1.1.115:/home/lzh/20260829/nuc/
```

### 3.2 Mac：同步三台 Jetson 的 UDP 代码和各自配置

下面的命令同样在 **Mac 的当前仓库终端**执行，不是在 NUC 或 Jetson 中执行：

```bash
UDP_SRC=/Users/steve/Documents/code/NetCatch/script/dynamics/nuc/udp

rsync -av --exclude 'config.json' "${UDP_SRC}/" orin1@10.1.1.110:/home/orin1/Documents/NetCatch/udp/
rsync -av "${UDP_SRC}/receiver-configs/receiver-config.uav1.json" orin1@10.1.1.110:/home/orin1/Documents/NetCatch/udp/receiver-config.json

rsync -av --exclude 'config.json' "${UDP_SRC}/" orin2@10.1.1.108:/home/orin2/Documents/NetCatch/udp/
rsync -av "${UDP_SRC}/receiver-configs/receiver-config.uav2.json" orin2@10.1.1.108:/home/orin2/Documents/NetCatch/udp/receiver-config.json

rsync -av --exclude 'config.json' "${UDP_SRC}/" orin3@10.1.1.101:/home/orin3/Documents/NetCatch/udp/
rsync -av "${UDP_SRC}/receiver-configs/receiver-config.uav3.json" orin3@10.1.1.101:/home/orin3/Documents/NetCatch/udp/receiver-config.json
```

三台 Jetson 的配置已经固定为：

| Jetson | `multicast_interface_ip` |
|---|---:|
| uav1 | `10.1.1.110` |
| uav2 | `10.1.1.108` |
| uav3 | `10.1.1.101` |

NUC 的 `udp/config.json` 应填写 NUC 自己的 IP：

```json
"multicast_interface_ip": "10.1.1.115"
```

只有设备 IP 改变时才需要修改。所有设备的组播参数保持 `239.255.42.99:15150`。

## 4. 每次实验的 UDP 启动步骤

### 4.1 NUC 终端 A：先启动 predictor

按照 [`../predict/README.md`](../predict/README.md) 操作。predictor 必须先发布 `/netcatch/dynamics/prediction`。

### 4.2 NUC 终端 B：启动唯一的 UDP sender

在 **NUC** 执行：

```bash
cd ~/20260829/nuc/udp
./run_udp_sender.sh ./config.json
```

正常日志应包含：

```text
UDP sender ready: topic=/netcatch/dynamics/prediction group=239.255.42.99:15150 interface=10.1.1.115 rate=30Hz
```

### 4.3 Jetson：诊断 receiver 可选，不必每次启动

只有需要观察网络状态时，才在任意一台 **Jetson 宿主机终端**执行：

```bash
docker exec -it rl_humble bash
```

进入容器后执行：

```bash
cd /workspace/udp
./run_udp_receiver.sh ./receiver-config.json --count 0 --timeout-s 5 --changes-only
```

按 `Ctrl-C` 停止。`--changes-only` 只在状态发生变化时输出，不会以 30 Hz 刷屏。三机通信已经验收后，不需要每次开机同时运行三份诊断 receiver。

## 5. 仅在没有数据时检查

先确认 NUC sender 仍在运行。随后可在 **NUC** 查看本机组播：

```bash
cd ~/20260829/nuc/udp
./run_udp_receiver.sh ./config.json --count 0 --timeout-s 5 --changes-only
```

如果需要抓包，在对应机器先用 `ip -4 -br addr` 确认接口名。当前 NUC 接口为 `wlo1`，Jetson 示例接口为 `wlP1p1s0`：

```bash
# 仅在 NUC 执行
sudo tcpdump -ni wlo1 'udp dst host 239.255.42.99 and dst port 15150'

# 仅在 Jetson 执行
sudo tcpdump -ni wlP1p1s0 'udp dst host 239.255.42.99 and dst port 15150'
```

正常时约每秒 30 个包。完整 JSON 只在确实需要检查协议字段时使用：

```bash
./run_udp_receiver.sh ./receiver-config.json --count 5 --timeout-s 5 --json
```

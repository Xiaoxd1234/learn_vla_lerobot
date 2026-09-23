# SO-101 学习与实践项目

这个目录面向 SO-101（leader + follower）和 Hugging Face LeRobot，覆盖硬件准备、标定、遥操作、数据采样、数据检查、ACT 训练、策略推理和安全控制。

## 获取仓库

`lerobot/` 和 `references/so101-real/` 以 Git 子模块固定到已验证版本。首次克隆时使用：

```powershell
git clone --recurse-submodules git@github.com:Xiaoxd1234/learn_vla_lerobot.git
cd learn_vla_lerobot
```

如果已经完成普通克隆，再初始化子模块：

```powershell
git submodule update --init --recursive
```

## 已准备内容

- `lerobot/`：官方 LeRobot 主线源码，来源为 `https://github.com/huggingface/lerobot`，当前快照 commit `240ea44cf9f887df944019cf46e18a0d496e1a56`。
- `references/course/`：从 `D:\TanZiyan` 筛选出的 SO-ARM101 使用、舵机固件和 BusLinker 文档。
- `references/local-workflow/`：本地已有的工作流、键盘控制和 GPU 测试脚本，仅作为参考，不直接覆盖官方实现。
- `references/so101-real/`：JereoZero 的 SO-101 真机模仿学习项目，包含 LeRobot 0.5.1/0.6.0 的校准、遥操作、采样、训练、推理和故障排查文档及脚本；详见该目录的 `VERSION_NOTICE.md`。
- `scripts/verify_env.ps1`：验证 Conda 环境、CUDA 和 LeRobot CLI。
- `environment.yml`：环境说明和依赖来源记录。

视频、3D 模型、数据集和训练权重不纳入版本控制：这些文件体积较大，并且数据集和权重与具体任务绑定。请在本地按任务存放，并保留原始数据备份。

## 环境

```powershell
conda activate so101-lerobot
cd <仓库目录>
powershell -ExecutionPolicy Bypass -File .\scripts\verify_env.ps1
```

环境已创建并验证：Python 3.12.13、PyTorch 2.11.0+cu128、RTX 4090、CUDA 可用；LeRobot 以 editable 方式连接到本目录源码。

## 学习顺序

1. **先理解接口**：阅读 `lerobot/docs/source/so101.mdx`、`docs/source/cheat-sheet.mdx` 和 `docs/source/il_robots.mdx`，理解 robot、teleop、camera、dataset、policy 五个对象。
2. **硬件和通信**：接好电源、USB 和电机总线；运行 `lerobot-find-port`。首次使用或更换电机才运行 `lerobot-setup-motors`，它会写入电机 ID/波特率。
3. **标定**：分别标定 follower 和 leader。标定文件会保存在用户目录，必须保留并与对应机械臂 ID 配对。
4. **无相机遥操作**：先确认 leader 能稳定控制 follower，再接相机。任何异常都先断电/急停，不要直接训练。
5. **相机和采样**：用 `lerobot-find-cameras` 确认索引；固定相机、分辨率和 FPS。每条 episode 只做一个清晰任务，先采 10 条检查，再扩展到 30--100 条。
6. **数据检查/回放**：用 `lerobot-dataset-viz` 看视频、状态和动作；用 `lerobot-replay` 在低速下回放，确认动作方向、夹爪和时间同步。
7. **第一次训练**：推荐 ACT，而不是一开始就训练 Pi0/SmolVLA。ACT 对 SO-101 小数据更容易调通，4090 可直接训练。先用 5--10k steps 做冒烟，再做 50k steps。
8. **推理和评估**：使用 `lerobot-rollout --policy.path=...`，先空载/旁观测试，再放入物体；至少记录 10 条成功率和失败原因。
9. **改进闭环**：只增加失败模式对应的数据；保持任务描述、相机配置、动作空间和数据版本一致。不要把不同任务混进同一个 `single_task` 数据集。

## Windows 命令模板

把下面的 `COM_FOLLOWER`、`COM_LEADER`、相机索引、任务名和路径替换成自己的值。命令使用 `--key=value` 形式。

### 端口、标定和遥操作

```powershell
lerobot-find-port
lerobot-calibrate --robot.type=so101_follower --robot.port=COM_FOLLOWER --robot.id=so101_follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM_LEADER --teleop.id=so101_leader
lerobot-teleoperate --robot.type=so101_follower --robot.port=COM_FOLLOWER --robot.id=so101_follower --teleop.type=so101_leader --teleop.port=COM_LEADER --teleop.id=so101_leader
```

### 相机和数据采样

```powershell
lerobot-find-cameras
lerobot-record `
  --robot.type=so101_follower --robot.port=COM_FOLLOWER --robot.id=so101_follower `
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
  --teleop.type=so101_leader --teleop.port=COM_LEADER --teleop.id=so101_leader `
  --dataset.repo_id=local/so101_pick --dataset.root=.\data\so101_pick `
  --dataset.single_task="Pick up the red block" --dataset.num_episodes=10 `
  --display_data=true
```

采样前先做 1 条测试 episode；确认画面、动作和任务文本正确后再批量采集。数据根目录建议放在 `so101-learning/data/`，不要提交 Git。

### 数据可视化与回放

```powershell
lerobot-dataset-viz --repo-id=local/so101_pick --root=.\data\so101_pick --episode-index=0
lerobot-replay --robot.type=so101_follower --robot.port=COM_FOLLOWER --robot.id=so101_follower --dataset.repo_id=local/so101_pick --dataset.root=.\data\so101_pick --dataset.episode=0 --dataset.fps=10
```

### ACT 训练

```powershell
lerobot-train `
  --policy.type=act `
  --dataset.repo_id=local/so101_pick --dataset.root=.\data\so101_pick `
  --output_dir=.\outputs\train\act_so101_pick --job_name=act_so101_pick `
  --policy.device=cuda --steps=10000 --batch_size=8 `
  --wandb.enable=false
```

训练前先确认数据至少有多个完整 episode。训练后重点查看 loss、动作是否平滑，以及验证 episode 的成功率；不要只根据 loss 判断策略可用性。

### 推理/控制

```powershell
lerobot-rollout `
  --strategy.type=base `
  --robot.type=so101_follower --robot.port=COM_FOLLOWER --robot.id=so101_follower `
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
  --policy.path=.\outputs\train\act_so101_pick\checkpoints\last\pretrained_model `
  --device=cuda --task="Pick up the red block" --duration=60 `
  --display_data=true --fps=10
```

推理时手边必须有断电/急停方式，先抬高机械臂、清空工作区、低速空载观察，再接触物体。策略控制的动作不是安全认证，任何卡滞、过流、撞击或异常发热都应立即停止。

## 模型选择建议

| 阶段 | 推荐 | 原因 |
| --- | --- | --- |
| 第一次闭环 | ACT | 数据量小、训练和部署简单，适合确认采样链路 |
| 动作更复杂 | Diffusion Policy | 对多模态动作更有表现力，但显存和调参成本更高 |
| 语言泛化 | SmolVLA / Pi0.5 | 需要更大模型、预训练权重和更多显存/网络下载，不作为第一步 |

## 官方资料和版本纪律

- 官方仓库：<https://github.com/huggingface/lerobot>
- SO-101 文档：`lerobot/docs/source/so101.mdx`
- CLI 速查：`lerobot/docs/source/cheat-sheet.mdx`
- 训练硬件建议：`lerobot/docs/source/hardware_guide.mdx`

本项目优先使用 `lerobot/` 里的命令和文档。`D:\TanZiyan\lerobot` 是旧版快照，里面的参数可能与 0.6.x 不一致；本地 `lerobot_workflow.py` 只作为排查思路参考。

### 外部真机案例参考

`references/so101-real/lerobot-0.6/` 是一个基于 LeRobot v0.6.0、Ubuntu/Linux 和特定 SO-101 配置整理的完整案例。建议重点阅读 `docs/01-hardware-setup.md` 至 `docs/07-troubleshooting.md`，学习其采样组织、数据质量检查和训练/推理排错过程；其中的实验记录（如任务成功率）是仓库作者在其硬件和数据上的结果，不代表本机效果。

本工作区使用 Windows 和 LeRobot v0.6.2 快照。因此该仓库仅作流程与设计参考，命令参数须以本地 `lerobot/` 为准。不要直接运行其中的 `apply_patches.py`：它会修改作者机器上的 LeRobot 源码，且 patch 假设不同版本。也不要直接照搬其 `fixed_joints`、夹爪扭矩、端口/相机配置或真机 rollout 命令；这些需要先结合你的硬件、官方当前实现和安全条件逐项验证。仓库没有在根目录提供许可证文件，复制来的内容仅用于本地学习参考；如需再分发或复用代码，先向仓库作者确认授权。

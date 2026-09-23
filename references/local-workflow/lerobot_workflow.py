"""LeRobot 工作流脚本：校准、遥操作、数据采集、训练、推理。

通过 subprocess 调用 lerobot 自带脚本，对应 new_测试代码.md 中的 5 个步骤。
关键参数全部定义在 `if __name__ == "__main__":` 中，便于统一修改。
注意：参数必须使用 --key=value 格式（带等号），这是 draccus 解析器的要求。
"""
import os
import subprocess
import sys

# 启用离线模式，使用本地缓存（避免网络连接超时）
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def run_cmd(cmd):
    """运行命令并打印。"""
    # 将列表转换为字符串命令，便于 shell 解析
    cmd_str = " ".join(cmd)
    print("执行命令:\n  " + cmd_str + "\n")
    subprocess.run(cmd_str, shell=True, check=True)


def calibrate_follower(robot_type, robot_port, robot_id):
    """1. 校准（从臂）"""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_calibrate",
        f"--robot.type={robot_type}",
        f"--robot.port={robot_port}",
        f"--robot.id={robot_id}",
    ]
    run_cmd(cmd)


def calibrate_leader(teleop_type, teleop_port, teleop_id):
    """1. 校准（主臂）"""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_calibrate",
        f"--teleop.type={teleop_type}",
        f"--teleop.port={teleop_port}",
        f"--teleop.id={teleop_id}",
    ]
    run_cmd(cmd)


def teleoperate(robot_type, robot_port, robot_id, teleop_type, teleop_port, teleop_id):
    """2. 遥操作（无相机）"""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_teleoperate",
        f"--robot.type={robot_type}",
        f"--robot.port={robot_port}",
        f"--robot.id={robot_id}",
        f"--teleop.type={teleop_type}",
        f"--teleop.port={teleop_port}",
        f"--teleop.id={teleop_id}",
    ]
    run_cmd(cmd)


def teleoperate_with_cameras(robot_type, robot_port, robot_id,
                             teleop_type, teleop_port, teleop_id,
                             cameras, display_data=True):
    """2. 遥操作（带相机）"""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_teleoperate",
        f"--robot.type={robot_type}",
        f"--robot.port={robot_port}",
        f"--robot.id={robot_id}",
        f"--robot.cameras=\"{cameras}\"",
        f"--teleop.type={teleop_type}",
        f"--teleop.port={teleop_port}",
        f"--teleop.id={teleop_id}",
        f"--display_data={str(display_data).lower()}",
    ]
    run_cmd(cmd)


def record(robot_type, robot_port, robot_id, cameras,
           teleop_type, teleop_port, teleop_id,
           dataset_root, dataset_repo_id, num_episodes, single_task,
           display_data=True, push_to_hub=False):
    """3. 数据采集"""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_record",
        f"--robot.type={robot_type}",
        f"--robot.port={robot_port}",
        f"--robot.id={robot_id}",
        f"--robot.cameras=\"{cameras}\"",
        f"--teleop.type={teleop_type}",
        f"--teleop.port={teleop_port}",
        f"--teleop.id={teleop_id}",
        f"--display_data={str(display_data).lower()}",
        f"--dataset.root=\"{dataset_root}\"",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.push_to_hub={str(push_to_hub).lower()}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task=\"{single_task}\"",
    ]
    run_cmd(cmd)


def train(dataset_repo_id, dataset_root, policy_type, output_dir, job_name, device,
          dataset_revision=None, policy_pretrained_path=None, compile_model=False,
          gradient_checkpointing=False, dtype="bfloat16", freeze_vision_encoder=False,
          train_expert_only=False, steps=50000, push_to_hub=False,
          wandb_enable=False, wandb_project=None, batch_size=8):
    """4. 训练"""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root=\"{dataset_root}\"",
        f"--policy.type={policy_type}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--policy.device={device}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--wandb.enable={str(wandb_enable).lower()}",
        f"--policy.push_to_hub={str(push_to_hub).lower()}",
    ]
    if dataset_revision:
        cmd.append(f"--dataset.revision={dataset_revision}")
    if policy_pretrained_path:
        cmd.append(f"--policy.pretrained_path={policy_pretrained_path}")
    if compile_model:
        cmd.append(f"--policy.compile_model={str(compile_model).lower()}")
    if gradient_checkpointing:
        cmd.append(f"--policy.gradient_checkpointing={str(gradient_checkpointing).lower()}")
    if dtype:
        cmd.append(f"--policy.dtype={dtype}")
    if freeze_vision_encoder:
        cmd.append(f"--policy.freeze_vision_encoder={str(freeze_vision_encoder).lower()}")
    if train_expert_only:
        cmd.append(f"--policy.train_expert_only={str(train_expert_only).lower()}")
    if wandb_project:
        cmd.append(f"--wandb.project={wandb_project}")
    run_cmd(cmd)


def rollout(robot_type, robot_port, robot_id, cameras,
            policy_path, device, strategy_type="base", fps=10.0, display_data=True,
            inference_type="sync", rtc_execution_horizon=10, rtc_max_guidance_weight=10.0):
    """5. 推理"""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_rollout",
        f"--robot.type={robot_type}",
        f"--robot.port={robot_port}",
        f"--robot.id={robot_id}",
        f"--robot.cameras=\"{cameras}\"",
        f"--display_data={str(display_data).lower()}",
        f"--policy.path=\"{policy_path}\"",
        f"--policy.device={device}",
        f"--strategy.type={strategy_type}",
        f"--fps={fps}",
        f"--inference.type={inference_type}",
    ]
    # 添加 RTC 参数(仅当 inference_type="rtc" 时生效)
    if inference_type == "rtc":
        cmd.append(f"--inference.rtc.execution_horizon={rtc_execution_horizon}")
        cmd.append(f"--inference.rtc.max_guidance_weight={rtc_max_guidance_weight}")
    run_cmd(cmd)


def replay_episode(robot_type, robot_port, robot_id,
                   dataset_repo_id, dataset_root, episode, fps=30, play_sounds=True):
    """回放指定 episode 的从动臂动作"""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_replay",
        f"--robot.type={robot_type}",
        f"--robot.port={robot_port}",
        f"--robot.id={robot_id}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.episode={episode}",
        f"--dataset.fps={fps}",
        f"--play_sounds={str(play_sounds).lower()}",
    ]
    if dataset_root:
        cmd.append(f"--dataset.root=\"{dataset_root}\"")
    run_cmd(cmd)


if __name__ == "__main__":

    # --- 机器人（从臂）---
    robot_type = "so101_follower"
    robot_port = "COM24"
    robot_id = "my_awesome_follower_arm"

    # --- 遥操作（主臂）---
    teleop_type = "so101_leader"
    teleop_port = "COM22"
    teleop_id = "my_awesome_leader_arm"

    # --- 相机配置（YAML 字符串，注意格式）---
    # 遥操作带相机
    cameras_teleop = "{ handeye: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"} }"
    # 数据采集
    cameras_record = "{ handeye: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"} }"
    # 推理
    # cameras_rollout = "{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"} }"
    cameras_rollout = cameras_record

    # --- 显示数据（可视化）---
    display_data = False

    # --- 数据集 ---
    dataset_root = "data/data_03"
    dataset_repo_id = "admin/demo"
    num_episodes = 30
    single_task = "Grab the screwdriver"
    # single_task = "Shake Hands"

    # --- 训练 ---
    policy_type = "pi05"
    output_dir = "outputs/train/pi05_so101"
    job_name = "pi05_so101"
    train_device = "cuda"
    dataset_revision = None  # 数据集版本，如 "v0.1.0"
    policy_pretrained_path = "pi0.5_base"
    compile_model = False  # 是否编译模型（Windows 不支持 Triton，需关闭）
    gradient_checkpointing = True  # 是否启用梯度检查点
    dtype = "bfloat16"  # 数据类型：float32, float16, bfloat16
    freeze_vision_encoder = True  # 是否冻结视觉编码器
    train_expert_only = True  # 是否只训练 expert
    steps = 50000  # 训练步数
    push_to_hub = False  # 是否推送到 Hub
    wandb_enable = False  # 是否启用 wandb（需先运行 wandb login 登录）
    wandb_project = "Lerobot_Pi05"  # wandb 项目名
    batch_size = 8  # 批次大小

    # --- 推理 ---
    # 注意：使用实际存在的 checkpoint 目录（如 100000），或训练完成后使用 last
    # policy_path = "outputs/train/act_so101_test/checkpoints/last/pretrained_model"
    # policy_path = "orange/orange/checkpoints/last/pretrained_model"
    # policy_path = "weight/ACT/30K/pretrained_model"
    # policy_path = "weight/pi0/50K/pretrained_model"
    policy_path = "outputs/train/pi05_so101/checkpoints/last/pretrained_model"

    rollout_device = "cuda"
    strategy_type = "base"
    rollout_fps = 30.0  # 推理目标 FPS（降低以匹配 PI05 推理速度）

    # --- RTC (Real-Time Chunking) 参数 ---
    # 用于慢速 VLA 模型(PI0/PI05)的推理优化
    # 启用 RTC 可以缓存多个动作,实现平滑控制
    inference_type = "sync"  # 推理类型: "sync" 或 "rtc"
    # 对于 PI0/PI05 模型,建议使用 "rtc"
    rtc_execution_horizon = 10  # 执行视野: 缓存多少个动作
    rtc_max_guidance_weight = 10.0  # 最大引导权重

    # --- 回放 ---
    replay_episode_index = 0  # 要回放的 episode 编号
    replay_fps = 30  # 回放帧率

    # ===== 选择要执行的任务 =====
    # 可选值:
    #   calibrate_follower      校准从臂
    #   calibrate_leader        校准主臂
    #   teleoperate             遥操作（无相机）
    #   teleoperate_with_cameras 遥操作（带相机）
    #   record                  数据采集
    #   replay_episode          回放指定 episode
    #   train                   训练
    #   rollout                 推理

    task = "rollout"

    if task == "calibrate_follower":
        calibrate_follower(robot_type, robot_port, robot_id)
    elif task == "calibrate_leader":
        calibrate_leader(teleop_type, teleop_port, teleop_id)
    elif task == "teleoperate":
        teleoperate(robot_type, robot_port, robot_id,
                    teleop_type, teleop_port, teleop_id)
    elif task == "teleoperate_with_cameras":
        teleoperate_with_cameras(robot_type, robot_port, robot_id,
                                 teleop_type, teleop_port, teleop_id,
                                 cameras_teleop, display_data)
    elif task == "record":
        record(robot_type, robot_port, robot_id, cameras_record,
               teleop_type, teleop_port, teleop_id,
               dataset_root, dataset_repo_id, num_episodes, single_task,
               display_data)
    elif task == "train":
        train(dataset_repo_id, dataset_root, policy_type,
              output_dir, job_name, train_device,
              dataset_revision, policy_pretrained_path, compile_model,
              gradient_checkpointing, dtype, freeze_vision_encoder,
              train_expert_only, steps, push_to_hub,
              wandb_enable, wandb_project, batch_size)
    elif task == "rollout":
        rollout(robot_type, robot_port, robot_id, cameras_rollout,
                policy_path, rollout_device, strategy_type, rollout_fps, display_data,
                inference_type, rtc_execution_horizon, rtc_max_guidance_weight)
    elif task == "replay_episode":
        replay_episode(robot_type, robot_port, robot_id,
                       dataset_repo_id, dataset_root, replay_episode_index, replay_fps)
    else:
        print(f"未知任务: {task}")

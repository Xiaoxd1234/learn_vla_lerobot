#!/usr/bin/env python
"""
使用 lerobot 封装的键盘末端执行器控制脚本

参考 lerobot 的标准架构，使用：
- KeyboardEndEffectorTeleop 读取键盘输入
- 自定义处理器流水线进行运动学转换
- SO101Follower 机械臂控制

使用方法：
```bash
python keyboard_control_6dof_lerobot.py
```
"""

import logging
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from pprint import pformat
from scipy.spatial.transform import Rotation

from lerobot.configs import parser
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
    ProcessorStep,
    ProcessorStepRegistry,
    PipelineFeatureType,
    PolicyFeature,
    EnvTransition,
    TransitionKey,
)
from lerobot.common.types import FeatureType
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun
from lerobot.utils.utils import init_logging


@ProcessorStepRegistry.register("keyboard_delta_to_ee_pose")
@dataclass
class KeyboardDeltaToEEPose(ProcessorStep):
    """
    将 KeyboardEndEffectorTeleop 的 delta 输出转换为末端执行器目标位姿

    输入: {"delta_x": -1/0/1, "delta_y": -1/0/1, "delta_z": -1/0/1, "gripper": 0/1/2}
    输出: {"ee.x": target_x, "ee.y": target_y, "ee.z": target_z, "ee.wx": ..., "gripper.pos": ...}
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    end_effector_step_sizes: dict
    _current_target: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """处理键盘 delta 动作"""
        self._current_transition = transition.copy()
        action = self.transition["action"]
        observation = self.transition["observation"]

        # 获取当前关节角度
        q_raw = np.array([observation[f"{motor}.pos"] for motor in self.motor_names])

        # 计算当前末端位姿（正向运动学）
        T_current = self.kinematics.forward_kinematics(q_raw)

        # 如果第一次运行，初始化目标为当前位置
        if self._current_target is None:
            self._current_target = T_current.copy()

        # 获取键盘 delta
        delta_x = action.get("delta_x", 0.0)
        delta_y = action.get("delta_y", 0.0)
        delta_z = action.get("delta_z", 0.0)
        gripper_action = action.get("gripper", 1.0)

        # 计算位置增量（转换为米）
        delta_pos = np.array([
            delta_x * self.end_effector_step_sizes["delta_x"],
            delta_y * self.end_effector_step_sizes["delta_y"],
            delta_z * self.end_effector_step_sizes["delta_z"],
        ])

        # 更新目标位置
        self._current_target[:3, 3] += delta_pos

        # 保持当前姿态不变
        # 提取目标位姿
        target_pos = self._current_target[:3, 3]
        target_rot = Rotation.from_matrix(self._current_target[:3, :3])
        target_rotvec = target_rot.as_rotvec()

        # 设置末端执行器目标位姿
        action["ee.x"] = float(target_pos[0])
        action["ee.y"] = float(target_pos[1])
        action["ee.z"] = float(target_pos[2])
        action["ee.wx"] = float(target_rotvec[0])
        action["ee.wy"] = float(target_rotvec[1])
        action["ee.wz"] = float(target_rotvec[2])

        # 处理夹爪（转换为关节位置）
        # gripper: 0=关闭, 1=保持, 2=打开
        if gripper_action == 2:  # 打开
            action["gripper.pos"] = observation.get("gripper.pos", 0) + 10
        elif gripper_action == 0:  # 关闭
            action["gripper.pos"] = observation.get("gripper.pos", 100) - 10
        else:  # 保持
            action["gripper.pos"] = observation.get("gripper.pos", 50)

        transition["action"] = action
        return transition

    def reset(self):
        """重置内部状态"""
        self._current_target = None


@ProcessorStepRegistry.register("ee_pose_to_joints")
@dataclass
class EEPoseToJoints(ProcessorStep):
    """
    将末端执行器目标位姿转换为关节角度（逆运动学）

    输入: {"ee.x": ..., "ee.y": ..., "ee.z": ..., "ee.wx": ..., "ee.wy": ..., "ee.wz": ..., "gripper.pos": ...}
    输出: {"shoulder_pan.pos": ..., "shoulder_lift.pos": ..., ..., "gripper.pos": ...}
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将末端位姿转换为关节角度"""
        self._current_transition = transition.copy()
        action = self.transition["action"]
        observation = self.transition["observation"]

        # 构建目标变换矩阵
        T_target = np.eye(4)

        # 提取目标位置
        target_pos = np.array([
            action.get("ee.x", 0.0),
            action.get("ee.y", 0.0),
            action.get("ee.z", 0.0),
        ])
        T_target[:3, 3] = target_pos

        # 提取目标姿态
        target_rotvec = np.array([
            action.get("ee.wx", 0.0),
            action.get("ee.wy", 0.0),
            action.get("ee.wz", 0.0),
        ])
        target_rot = Rotation.from_rotvec(target_rotvec)
        T_target[:3, :3] = target_rot.as_matrix()

        # 获取当前关节角度（作为初始猜测）
        current_joints = np.array([observation[f"{motor}.pos"] for motor in self.motor_names])

        # 使用逆运动学求解目标关节角度
        try:
            target_joints = self.kinematics.inverse_kinematics(
                current_joint_pos=current_joints,
                desired_ee_pose=T_target,
                position_weight=1.0,
                orientation_weight=0.1,  # 轻微约束姿态
            )

            # 设置关节目标
            for i, motor in enumerate(self.motor_names):
                action[f"{motor}.pos"] = float(target_joints[i])

            logger = logging.getLogger(__name__)
            logger.debug(f"IK solved: pos error={np.linalg.norm(target_pos - T_target[:3, 3])*1000:.2f}mm")

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"IK solving failed: {e}")
            # 如果 IK 失败，保持当前位置
            for motor in self.motor_names:
                action[f"{motor}.pos"] = observation[f"{motor}.pos"]

        transition["action"] = action
        return transition

    def reset(self):
        """重置内部状态"""
        pass


@dataclass
class Keyboard6DOFConfig:
    """键盘6自由度控制配置"""

    # ========== 机械臂配置（根据您的设备修改）==========
    robot_port: str = "/dev/ttyACM0"  # Windows: "COM3", Linux: "/dev/ttyACM0"
    robot_id: str = "my_awesome_follower_arm"  # 必须与校准文件中的 robot_id 一致
    use_degrees: bool = True

    # ========== 校准文件配置 ==========
    # 校准文件路径：/home/tzy/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json
    calibration_dir: str = "/home/tzy/.cache/huggingface/lerobot/calibration/robots/so_follower"

    # ========== URDF路径配置 ==========
    urdf_path: str = str(Path(__file__).parent.parent / "SO101" / "so101_new_calib.urdf")

    # ========== 控制参数 ==========
    fps: int = 30  # 控制频率
    display_data: bool = False  # 是否显示可视化数据

    # ========== 运动学参数（每次按键移动的距离）==========
    end_effector_step_sizes: dict = field(default_factory=lambda: {
        "delta_x": 0.002,  # 2mm
        "delta_y": 0.002,  # 2mm
        "delta_z": 0.002,  # 2mm
        "gripper": 10.0,   # 夹爪单位
    })


def create_keyboard_ee_pipeline(kinematics: RobotKinematics, motor_names: list[str], step_sizes: dict):
    """
    创建键盘末端执行器控制处理器流水线

    流程：
    1. KeyboardEndEffectorTeleop -> delta_x, delta_y, delta_z (相对增量)
    2. KeyboardDeltaToEEPose -> 绝对目标位置 (使用正向运动学计算当前位置)
    3. EEPoseToJoints -> 目标关节角度 (使用逆向运动学)
    """
    pipeline = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            # 步骤1: 将键盘 delta 转换为末端执行器目标位姿
            KeyboardDeltaToEEPose(
                kinematics=kinematics,
                motor_names=motor_names,
                end_effector_step_sizes=step_sizes,
            ),
            # 步骤2: 将末端执行器目标位姿转换为关节角度
            EEPoseToJoints(
                kinematics=kinematics,
                motor_names=motor_names,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return pipeline


@parser.wrap()
def keyboard_control_6dof(cfg: Keyboard6DOFConfig):
    """
    使用 lerobot 封装的键盘末端执行器控制

    支持命令行参数，类似 lerobot-teleoperate 命令
    """
    init_logging()
    logger = logging.getLogger(__name__)
    logger.info(pformat(cfg))

    # 创建机械臂配置（包含校准文件路径）
    from pathlib import Path as PathlibPath

    calibration_dir = PathlibPath(cfg.calibration_dir) if cfg.calibration_dir else None

    robot_config = SO101FollowerConfig(
        port=cfg.robot_port,
        id=cfg.robot_id,
        use_degrees=cfg.use_degrees,
        disable_torque_on_disconnect=True,
        calibration_dir=calibration_dir,  # 添加校准文件目录
    )

    # 初始化机械臂
    logger.info("初始化机械臂...")
    robot = SO101Follower(robot_config)

    # 初始化运动学求解器
    logger.info(f"加载 URDF: {cfg.urdf_path}")
    motor_names = list(robot.bus.motors.keys())[:-1]  # 去掉 gripper
    kinematics = RobotKinematics(
        urdf_path=cfg.urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )

    # 初始化键盘遥操作
    logger.info("初始化键盘遥操作...")
    teleop_config = KeyboardEndEffectorTeleopConfig(use_gripper=True)
    teleop = KeyboardEndEffectorTeleop(teleop_config)

    # 创建处理器流水线
    logger.info("创建处理器流水线...")
    keyboard_to_joints_pipeline = create_keyboard_ee_pipeline(
        kinematics=kinematics,
        motor_names=motor_names,
        step_sizes=cfg.end_effector_step_sizes,
    )

    # 连接设备（使用已有校准，避免重复校准）
    logger.info("连接设备...")
    robot.connect(calibrate=False)  # 使用已有的校准文件
    teleop.connect()

    # 初始化可视化
    if cfg.display_data:
        init_rerun(session_name="keyboard_6dof_control")

    print("\n" + "=" * 60)
    print("SO-101 机械臂键盘末端执行器控制 (lerobot封装版)")
    print("=" * 60)
    print(f"机械臂端口: {cfg.robot_port}")
    print(f"URDF路径: {cfg.urdf_path}")
    print(f"控制频率: {cfg.fps} Hz")
    print("=" * 60)
    print("\n位置控制:")
    print("  ↑/↓ 或 W/S: 前后移动 (每次2mm)")
    print("  ←/→ 或 A/D: 左右移动 (每次2mm)")
    print("  Shift_L/Shift_R: 上下移动 (每次2mm)")
    print("\n夹爪控制:")
    print("  Ctrl_R: 打开夹爪")
    print("  Ctrl_L: 关闭夹爪")
    print("\n其他:")
    print("  ESC: 退出")
    print("=" * 60)

    try:
        logger.info("开始控制循环...")
        while teleop.is_connected:
            loop_start = time.perf_counter()

            # 获取机械臂观测
            obs = robot.get_observation()

            # 获取键盘动作 (delta_x, delta_y, delta_z, gripper)
            teleop_action = teleop.get_action()

            # 通过处理器流水线：delta -> 绝对位置 -> 关节角度
            robot_action = keyboard_to_joints_pipeline((teleop_action, obs))

            # 发送关节角度给机械臂
            robot.send_action(robot_action)

            # 可视化
            if cfg.display_data:
                log_rerun_data(observation=obs, action=robot_action)

            # 控制频率
            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1.0 / cfg.fps - dt_s, 0.0))

    except KeyboardInterrupt:
        logger.info("用户中断")

    finally:
        if cfg.display_data:
            shutdown_rerun()
        teleop.disconnect()
        robot.disconnect()
        logger.info("程序结束")


def main():
    keyboard_control_6dof()


if __name__ == "__main__":
    main()
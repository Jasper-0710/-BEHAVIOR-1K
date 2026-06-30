"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""

import os
import time
import torch as th
import numpy as np
import threading
import queue  # [新增] 用于实现多线程异步保存队列

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
import omnigibson.utils.transform_utils as T
from lerobot.datasets.lerobot_dataset import LeRobotDataset


CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    office_large="office large",
    school_chemistry="school chemistry",
    Ihlen_1_int = "Ihlen 1 int",
    empty="Empty environment with no objects",
)

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


class LeRobotRecorder:
    """LeRobot v2.1 格式数据录制器 - 异步后台保存 + 丢弃功能版"""
    
    def __init__(self, robot, fps=30, sim_fps=60):
        self.robot = robot
        self.is_recording = False
        self.fps = fps  
        self.sim_fps = sim_fps  
        self.dataset = None
        
        timestamp = int(time.time())
        self.repo_id = f"omnigibson_robot_{timestamp}"
        self.local_dir = os.path.abspath(f"./lerobot_data/{self.repo_id}")
        self.task_description = "Robot teleoperation task"
        
        self.action_dim = 24
        self.state_dim = 24
        
        self.episode_buffer = []
        self.saved_episode_count = 0
        
        self.frame_interval = 1.0 / fps  
        self.last_record_time = 0.0  
        
        self.camera_names = self._detect_cameras()

        # ================= [新增] 多线程异步保存队列配置 =================
        self.save_queue = queue.Queue(maxsize=10)
        # 创建并启动后台保存线程，daemon=True 保证主线程崩溃时它也会自动退出
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        # =============================================================
    
    def _detect_cameras(self):
        cameras = {}
        if hasattr(self.robot, 'sensors'):
            for sensor_name, sensor in self.robot.sensors.items():
                from omnigibson.sensors import VisionSensor
                if isinstance(sensor, VisionSensor):
                    sensor_lower = sensor_name.lower()
                    if any(keyword in sensor_lower for keyword in ['eyes', 'head', 'zed']):
                        cameras['head'] = sensor_name
                    elif any(keyword in sensor_lower for keyword in ['eef', 'wrist', 'hand', 'gripper', 'realsense']):
                        if 'left' in sensor_lower:
                            cameras['left_wrist'] = sensor_name
                        elif 'right' in sensor_lower:
                            cameras['right_wrist'] = sensor_name
                        else:
                            cameras['wrist'] = sensor_name
        
        if cameras:
            print(f"📷 检测到 {len(cameras)} 个摄像头: {list(cameras.values())}")
        else:
            print("⚠️ 未检测到任何摄像头！")
        return cameras
    
    # ================= [新增] 初始化数据集的内部函数 =================
    def _init_dataset_if_needed(self):
        """延迟初始化 Dataset，确保它在后台线程中被创建，避免多线程冲突"""
        if self.dataset is None:
            print(f"📁 正在后台创建新的 LeRobot 数据集 (路径: {self.local_dir})...")
            features = {
                "observation.state": {"dtype": "float32", "shape": (self.state_dim,), "names": ["dim"]},
                "action": {"dtype": "float32", "shape": (self.action_dim,), "names": ["dim"]}
            }
            for cam_name in self.camera_names.keys():
                features[f"observation.image.{cam_name}"] = {
                    "dtype": "video", "shape": (3, 512, 512), "names": ["c", "h", "w"]
                }
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                root=self.local_dir,
                features=features
            )
    # =============================================================

    # ================= [新增] 后台保存工作线程 =================
    def _save_worker(self):
        """后台消费队列中的 Episode 数据并保存到磁盘"""
        while True:
            episode_data = self.save_queue.get()
            
            # 如果收到 None，说明要求退出线程
            if episode_data is None:
                self.save_queue.task_done()
                break
                
            # 确保数据集对象已创建
            self._init_dataset_if_needed()
            
            # 提取数据并开始保存
            frames = episode_data['frames']
            print(f"\n📦 [后台保存] 开始处理 1 个包含 {len(frames)} 帧的 Episode...")
            
            for frame in frames:
                self.dataset.add_frame(frame, self.task_description)
            
            self.dataset.save_episode()
            self.saved_episode_count += 1
            print(f"✅ [后台保存] 第 {self.saved_episode_count} 回合已成功写入磁盘！(剩余待保存队列: {self.save_queue.qsize()})")
            
            # 标记该任务完成
            self.save_queue.task_done()
    # =============================================================

    def start_recording(self):
        if self.is_recording:
            print("⚠️ [B] 当前正在录制中！请先按 [S] 保存 或 [Z] 丢弃。")
            return
        self.is_recording = True
        self.episode_buffer = []
        self.last_record_time = time.time()
        print("\n🔴 [B] 开始新回合录制！(数据暂存至内存...)")
    
    # ================= [新增] 丢弃当前回合函数 =================
    def discard_episode(self):
        if not self.is_recording:
            print("⚠️ [Z] 当前未在录制，无数据可丢弃！")
            return
        self.is_recording = False
        discard_len = len(self.episode_buffer)
        self.episode_buffer = []
        print(f"\n🗑️ [Z] 当前回合已丢弃 (清空 {discard_len} 帧)！你可以随时按 [B] 重新开始录制。")
    # =============================================================

    # ================= [修改] 异步保存逻辑 =================
    def stop_and_save_episode(self):
        if not self.is_recording:
            print("⚠️ [S] 当前未在录制！")
            return
        
        self.is_recording = False
        
        if not self.episode_buffer:
            print("⚠️ [S] 本次录制无有效帧，跳过保存。")
            return
        
        print(f"\n⏹️ [S] 录制停止！已将 {len(self.episode_buffer)} 帧推入后台保存队列。")
        print("💡 你现在可以立即按 [B] 开始下一回合的录制了！")
        
        # 将数据包装并放入后台队列，主线程立刻解脱
        self.save_queue.put({'frames': self.episode_buffer})
        
        # 清空当前缓冲，为下一次录制做准备
        self.episode_buffer = []
    # =============================================================
    
    # ================= [修改] 退出逻辑：等待后台线程 =================
    def finalize_and_exit(self):
        if self.is_recording and self.episode_buffer:
            print("\n⚠️ 注意: 当前仍在录制中，你有未处理的数据！")
            print("💡 建议: 先按 [S] 保存 或 [Z] 丢弃当前回合，然后再按 [F] 退出")
            return False
            
        print(f"\n{'='*80}")
        print(f"🔍 正在关闭程序...")
        
        # 如果队列中还有没保存完的数据，等待其完成
        if not self.save_queue.empty():
            print(f"⏳ 正在等待后台保存队列中的数据写入磁盘 (剩余 {self.save_queue.qsize()} 个 Episode)...")
        
        # 发送结束信号并等待线程结束
        self.save_queue.put(None)
        self.save_thread.join()
        
        print(f"\n🎉 统计信息:")
        print(f"   - 总计保存回合数: {self.saved_episode_count}")
        print(f"   - 数据集路径: {self.local_dir}")
        print(f"{'='*80}")
        return True
    # =============================================================
    
    def step(self, obs_dict, state, action):
        if not self.is_recording:
            return
        current_time = time.time()
        time_since_last_record = current_time - self.last_record_time
        if time_since_last_record < self.frame_interval:
            return
        self.last_record_time = current_time
        
        def to_numpy_array(x):
            if hasattr(x, 'detach'):
                return x.detach().cpu().numpy().astype(np.float32)
            else:
                return x.astype(np.float32)
        
        state_np = to_numpy_array(state)
        action_np = to_numpy_array(action)
        
        frame_data = {
            "observation.state": th.from_numpy(state_np).clone(),
            "action": th.from_numpy(action_np).clone(),
        }
        
        for cam_name, sensor_name in self.camera_names.items():
            rgb_img = None
            if sensor_name in obs_dict and isinstance(obs_dict[sensor_name], dict):
                if 'rgb' in obs_dict[sensor_name]:
                    rgb_img = obs_dict[sensor_name]['rgb']
            if rgb_img is None:
                robot_prefix = sensor_name.split(':')[0] if ':' in sensor_name else sensor_name
                if robot_prefix in obs_dict and isinstance(obs_dict[robot_prefix], dict):
                    if sensor_name in obs_dict[robot_prefix] and isinstance(obs_dict[robot_prefix][sensor_name], dict):
                        if 'rgb' in obs_dict[robot_prefix][sensor_name]:
                            rgb_img = obs_dict[robot_prefix][sensor_name]['rgb']
            if rgb_img is None:
                def find_rgb_in_dict(d, target_key, depth=0):
                    if depth > 3: return None
                    if isinstance(d, dict):
                        if target_key in d and isinstance(d[target_key], dict) and 'rgb' in d[target_key]:
                            return d[target_key]['rgb']
                        for k, v in d.items():
                            result = find_rgb_in_dict(v, target_key, depth + 1)
                            if result is not None: return result
                    return None
                rgb_img = find_rgb_in_dict(obs_dict, sensor_name)
            
            if rgb_img is not None:
                if hasattr(rgb_img, 'detach'):
                    rgb_img = rgb_img.detach().cpu().numpy()
                if rgb_img.shape[-1] == 4:
                    rgb_img = rgb_img[:, :, :3]
                img_chw = np.transpose(rgb_img, (2, 0, 1)).astype(np.uint8)
                frame_data[f"observation.image.{cam_name}"] = th.from_numpy(img_chw).clone()
        
        has_image = any(key.startswith("observation.image.") for key in frame_data.keys())
        if has_image:
            self.episode_buffer.append(frame_data)
        else:
            if len(self.episode_buffer) == 0:
                print(f"\n❌ 错误: 没有任何摄像头图像被采集！")

def choose_controllers(robot, random_selection=False):
    controller_choices = dict()
    default_config = robot._default_controller_config
    controller_names = robot.controller_order
    for controller_name in controller_names:
        controller_options = default_config[controller_name]
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options,
            name=f"{controller_name} controller",
            random_selection=random_selection,
        )
        controller_choices[controller_name] = choice
    return controller_choices

def get_24d_state(robot):
    """
    将机器人的 27 维状态转换为录制需要的 24 维状态。
    """
    joint_positions = robot.get_joint_positions()
    joint_velocities = robot.get_joint_velocities()
    _, orn = robot.get_position_orientation()
    rot_mat = T.quat2mat(orn)
    
    world_lin_vel = joint_velocities[:3]
    local_lin_vel = rot_mat.T @ world_lin_vel 
    
    world_ang_vel = robot.get_angular_velocity()
    local_ang_vel = rot_mat.T @ world_ang_vel
    
    local_vx = local_lin_vel[0]
    local_vy = local_lin_vel[1]
    local_rz_vel = local_ang_vel[2] 
    base_state_3d = th.stack([local_vx, local_vy, local_rz_vel])
    
    arm_and_gripper_positions = joint_positions[6:27]
    state_24d = th.cat([base_state_3d, arm_and_gripper_positions])
    
    return state_24d


def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
    scene_model = "empty"
    if not quickstart:
        scene_model = choose_from_options(options=SCENES, name="scene", random_selection=random_selection)

    robot_name = "Fetch"
    if not quickstart:
        robot_name = choose_from_options(
            options=list(sorted(REGISTERED_ROBOTS.keys())), name="robot", random_selection=random_selection
        )

    scene_cfg = dict()
    if scene_model == "empty":
        scene_cfg["type"] = "Scene"
    else:
        scene_cfg["type"] = "InteractiveTraversableScene"
        scene_cfg["scene_model"] = scene_model

    robot0_cfg = dict()
    robot0_cfg["type"] = robot_name
    robot0_cfg["obs_modalities"] = ["rgb"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = False
    robot0_cfg["sensor_config"] = {
        "VisionSensor": {
            "sensor_kwargs": {
                "image_width": 512,
                "image_height": 512
            }
        }
    }

    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])
    env = og.Environment(configs=cfg)

    robot = env.robots[0]
    controller_choices = {
        "base": "DifferentialDriveController",
        "arm_0": "InverseKinematicsController",
        "gripper_0": "MultiFingerGripperController",
        "camera": "JointController",
    }
    if not quickstart:
        controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    if random_selection:
        control_mode = "random"
    elif quickstart:
        control_mode = "teleop"
    else:
        control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")

    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)
    env.scene.update_initial_file()

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.46949, -3.97358, 2.21529]),
        orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    env.reset()
    robot.reset()

    action_generator = KeyboardRobotController(robot=robot)
    recorder = LeRobotRecorder(robot=robot, fps=30, sim_fps=60)

    flags = {"reset": False, "quit": False}
    
    def set_reset(): flags["reset"] = True
    def set_quit(): flags["quit"] = True

    # 注册控制按键
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.B,
        description="Start recording (B)",
        callback_fn=lambda: recorder.start_recording(),
    )
    
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.S,
        description="Save episode (S)",
        callback_fn=lambda: recorder.stop_and_save_episode(),
    )

    # ================= [新增] 注册 Z 键用于丢弃 =================
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.Z,
        description="Discard current episode (Z)",
        callback_fn=lambda: recorder.discard_episode(),
    )
    # ==========================================================
    
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset env (R)",
        callback_fn=set_reset,
    )
    
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.F,
        description="Finalize and exit (F)",
        callback_fn=set_quit,
    )

    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # ================= [修改] 控制台说明 UI =================
    print("\n" + "="*80)
    print("📹 LeRobot 数据录制控制说明（异步保存加强版）:")
    print("="*80)
    print("  [B] - 开始一个新回合的录制 (Begin)")
    print("  [Z] - 丢弃当前回合录制的所有数据 (Discard)  <-- NEW")
    print("  [S] - 停止录制并放入后台保存队列 (Save)  <-- NEW (不会阻塞，按完可立刻按B继续)")
    print("  [R] - 重置仿真环境 (Reset)")
    print("  [F] - 安全退出录制程序并等待数据落盘 (Finish)")
    print("="*80)
    print("💡 使用流程建议:")
    print("  1. 按 [B] 开始 -> 操作成功 -> 按 [S] 保存 -> 立刻按 [B] 开始下一轮")
    print("  2. 按 [B] 开始 -> 操作失误 -> 按 [Z] 丢弃 -> 立刻按 [B] 重新录制")
    print("="*80 + "\n")

    max_steps = -1 if not short_exec else 100
    step = 0

    action = robot.get_joint_positions()
    obs, _, _, _, _ = env.step(action=action)
    random_action = None
    
    while step != max_steps:
        if flags["quit"]:
            if recorder.finalize_and_exit():
                og.shutdown()
                break
            else:
                flags["quit"] = False
        
        if flags["reset"]:
            env.reset()
            robot.reset()
            flags["reset"] = False
            continue
        
        if control_mode == "random":
            if step % 30 == 0:
                random_action = action_generator.get_random_action() * 0.05
            action = random_action
        else:
            action = action_generator.get_teleop_action()
            action[:3] = action[:3] * 2
        
        state_24d = get_24d_state(robot)
        
        # print("*" *100)
        # print(state_24d)
        # print(action)
        # print("*" *100)
        
        next_obs, _, _, _, _ = env.step(action=action)
        
        # next_state_24d = get_24d_state(robot)
        
        if recorder.is_recording:
            recorder.step(obs_dict=obs, state=state_24d, action=action)
            if len(recorder.episode_buffer) > 0 and len(recorder.episode_buffer) % 30 == 0:
                elapsed_time = time.time() - recorder.last_record_time + (len(recorder.episode_buffer) * recorder.frame_interval)
                actual_fps = len(recorder.episode_buffer) / elapsed_time if elapsed_time > 0 else 0
                print(f"📹 录制中... 已采集 {len(recorder.episode_buffer)} 帧 (实际FPS: {actual_fps:.1f})", end='\r')
        
        obs = next_obs
        step += 1

    if not flags["quit"]:
        og.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Teleoperate a robot in a BEHAVIOR scene.")
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Whether the example should be loaded with default settings for a quick start.",
    )
    args = parser.parse_args()
    main(quickstart=args.quickstart)
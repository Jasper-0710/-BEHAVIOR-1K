import argparse
import json
import math
import os
import time
from collections import deque
from types import MethodType

import torch as th
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, ActionPrimitiveErrorGroup
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.robots.tiago import Tiago
from omnigibson.utils.geometry_utils import wrap_angle
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import KeyboardEventHandler


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DATASET_SCENES_DIR = os.path.join(REPO_ROOT, "datasets", "behavior-1k-assets", "scenes")
DEFAULT_SCENE_MODEL = "Pomaria_2_int_data_collection_3"
STANDBY_STEPS = 90
LOOK_AT_STEPS = 90
FACE_TARGET_STEPS = 120
FACE_TARGET_MAX_YAW_STEP = 0.018
MAX_CLICK_PLACE_DISTANCE = 10
PRIMITIVE_ATTEMPTS = 2
PRINT_NAV_CANDIDATE_DEBUG = True
MAX_NAV_CANDIDATE_DEBUG_LINES = 12
NAV_GOOD_SCORE_THRESHOLD = 0.35
NAV_MIN_VALID_CANDIDATES_FOR_EARLY_STOP = 8


def install_viewer_camera_without_rgb_annotator():
    """
    Work around Isaac / Replicator crashes when OmniGibson's viewer camera attaches RGB annotators.

    This only affects the UI viewer camera at /viewer_camera. Robot cameras used for LeRobot recording
    still keep their RGB modality from tiago_primitives.yaml.
    """
    from omnigibson.sensors import VisionSensor

    if getattr(VisionSensor, "_record_demo_viewer_camera_patch", False):
        return

    original_initialize_sensors = VisionSensor.initialize_sensors

    def initialize_sensors_skip_viewer_rgb(self, names):
        if self.name == "viewer_camera" or self.prim_path.endswith("/viewer_camera"):
            self._modalities = set()
            self._annotators = {}
            print("[viewer] Skipped RGB annotator for /viewer_camera.")
            return
        return original_initialize_sensors(self, names)

    VisionSensor.initialize_sensors = initialize_sensors_skip_viewer_rgb
    VisionSensor._record_demo_viewer_camera_patch = True
    print("[viewer] Patched /viewer_camera to skip RGB annotator; robot RGB cameras are unchanged.")


class LeRobotRecorder:
    """LeRobot 格式数据录制器；B 开始，S 保存，Z 丢弃，F 退出前等待落盘。"""

    def __init__(self, robot, fps=30):
        self.robot = robot
        self.is_recording = False
        self.fps = fps
        self.dataset = None

        timestamp = int(time.time())
        self.repo_id = f"omnigibson_task_{timestamp}"
        self.local_dir = os.path.abspath(f"./lerobot_data/{self.repo_id}")
        self.task_description = "OmniGibson pick and place task"

        self.action_dim = int(robot.action_dim)
        self.state_dim = 24
        self.episode_buffer = []
        self.saved_episode_count = 0
        self.frame_interval = 1.0 / fps
        self.last_record_time = 0.0
        self._finalized = False

        import queue
        import threading

        self.camera_shapes = {}
        self.camera_names = self._detect_cameras()
        self.save_queue = queue.Queue(maxsize=10)
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

    def _detect_cameras(self):
        cameras = {}
        if hasattr(self.robot, "sensors"):
            for sensor_name, sensor in self.robot.sensors.items():
                from omnigibson.sensors import VisionSensor

                if not isinstance(sensor, VisionSensor):
                    continue
                self.camera_shapes[sensor_name] = (3, int(sensor.image_height), int(sensor.image_width))
                sensor_lower = sensor_name.lower()
                if any(keyword in sensor_lower for keyword in ("eyes", "head", "zed")):
                    cameras["head"] = sensor_name
                elif any(keyword in sensor_lower for keyword in ("eef", "wrist", "hand", "gripper", "realsense")):
                    if "left" in sensor_lower:
                        cameras["left_wrist"] = sensor_name
                    elif "right" in sensor_lower:
                        cameras["right_wrist"] = sensor_name
                    else:
                        cameras["wrist"] = sensor_name

        if cameras:
            print(f"[record] detected cameras: {cameras}")
        else:
            print("[record-warn] no robot RGB camera detected; frames will not be saved until images are available.")
        return cameras

    def _init_dataset_if_needed(self):
        if self.dataset is not None:
            return

        print(f"[record] creating LeRobot dataset: {self.local_dir}")
        features = {
            "observation.state": {"dtype": "float32", "shape": (self.state_dim,), "names": ["dim"]},
            "action": {"dtype": "float32", "shape": (self.action_dim,), "names": ["dim"]},
        }
        for cam_name in self.camera_names:
            sensor_name = self.camera_names[cam_name]
            features[f"observation.image.{cam_name}"] = {
                "dtype": "video",
                "shape": self.camera_shapes.get(sensor_name, (3, 128, 128)),
                "names": ["c", "h", "w"],
            }
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset.create(repo_id=self.repo_id, fps=self.fps, root=self.local_dir, features=features)

    def _save_worker(self):
        while True:
            episode_data = self.save_queue.get()
            if episode_data is None:
                self.save_queue.task_done()
                break

            self._init_dataset_if_needed()
            frames = episode_data["frames"]
            print(f"\n[record] saving episode with {len(frames)} frames...")
            for frame in frames:
                self.dataset.add_frame(frame, self.task_description)
            self.dataset.save_episode()
            self.saved_episode_count += 1
            print(f"[record] saved episode {self.saved_episode_count}; queue={self.save_queue.qsize()}")
            self.save_queue.task_done()

    def start_recording(self):
        if self.is_recording:
            print("[record-warn] already recording; press S to save or Z to discard first.")
            return
        self.is_recording = True
        self.episode_buffer = []
        self.last_record_time = time.time()
        print("\n[record] started episode")

    def discard_episode(self):
        if not self.is_recording:
            print("[record-warn] not recording; nothing to discard.")
            return
        discard_len = len(self.episode_buffer)
        self.is_recording = False
        self.episode_buffer = []
        print(f"\n[record] discarded episode with {discard_len} frames")

    def stop_and_save_episode(self):
        if not self.is_recording:
            print("[record-warn] not recording.")
            return

        self.is_recording = False
        if not self.episode_buffer:
            print("[record-warn] empty episode; skip saving.")
            return

        print(f"\n[record] queued episode with {len(self.episode_buffer)} frames for background save")
        self.save_queue.put({"frames": self.episode_buffer})
        self.episode_buffer = []

    def finalize_and_exit(self):
        if self._finalized:
            return True
        if self.is_recording and self.episode_buffer:
            print("[record-warn] still recording; press S to save or Z to discard before exiting.")
            return False

        if not self.save_queue.empty():
            print(f"[record] waiting for save queue: {self.save_queue.qsize()} episode(s)")
        self.save_queue.put(None)
        self.save_thread.join()
        self._finalized = True
        print(f"[record] finalized. saved episodes={self.saved_episode_count}, path={self.local_dir}")
        return True

    def step(self, obs_dict, state, action):
        if not self.is_recording:
            return

        current_time = time.time()
        if current_time - self.last_record_time < self.frame_interval:
            return
        self.last_record_time = current_time

        import numpy as np

        state_np = state.detach().cpu().numpy().astype(np.float32) if hasattr(state, "detach") else state.astype(np.float32)
        action_np = (
            action.detach().cpu().numpy().astype(np.float32) if hasattr(action, "detach") else action.astype(np.float32)
        )
        frame_data = {
            "observation.state": th.from_numpy(state_np).clone(),
            "action": th.from_numpy(action_np).clone(),
        }

        for cam_name, sensor_name in self.camera_names.items():
            rgb_img = self._find_rgb(obs_dict, sensor_name)
            if rgb_img is None:
                continue
            if hasattr(rgb_img, "detach"):
                rgb_img = rgb_img.detach().cpu().numpy()
            if rgb_img.shape[-1] == 4:
                rgb_img = rgb_img[:, :, :3]
            img_chw = np.transpose(rgb_img, (2, 0, 1)).astype(np.uint8)
            frame_data[f"observation.image.{cam_name}"] = th.from_numpy(img_chw).clone()

        if any(key.startswith("observation.image.") for key in frame_data):
            self.episode_buffer.append(frame_data)
        elif len(self.episode_buffer) == 0:
            print("[record-error] no camera image found in observation; episode has no frames yet.")

    def _find_rgb(self, obs_dict, sensor_name):
        if sensor_name in obs_dict and isinstance(obs_dict[sensor_name], dict) and "rgb" in obs_dict[sensor_name]:
            return obs_dict[sensor_name]["rgb"]

        def recurse(value, depth=0):
            if depth > 4 or not isinstance(value, dict):
                return None
            if sensor_name in value and isinstance(value[sensor_name], dict) and "rgb" in value[sensor_name]:
                return value[sensor_name]["rgb"]
            for child in value.values():
                result = recurse(child, depth + 1)
                if result is not None:
                    return result
            return None

        return recurse(obs_dict)


def get_24d_state(robot):
    """记录用状态：base 局部速度 3 维 + Tiago 上半身/双臂/夹爪关节 21 维。"""
    joint_positions = robot.get_joint_positions()
    joint_velocities = robot.get_joint_velocities()
    _, orn = robot.get_position_orientation()
    rot_mat = T.quat2mat(orn)

    local_lin_vel = rot_mat.T @ joint_velocities[:3]
    local_ang_vel = rot_mat.T @ robot.get_angular_velocity()
    base_state_3d = th.stack([local_lin_vel[0], local_lin_vel[1], local_ang_vel[2]])
    arm_and_gripper_positions = joint_positions[6:27]
    return th.cat([base_state_3d, arm_and_gripper_positions])


def hold_current_pose_action(robot, controller=None):
    """生成一个保持当前关节位置的 action；传入 controller 时会顺带执行本 demo 的头部相机追踪。"""
    action = robot.q_to_action(robot.get_joint_positions())
    return apply_demo_head_tracking(robot, controller, action) if controller is not None else action


def overwrite_head_action_clamped_to_position(robot, action, target_pos):
    """
    交互 demo 专用的头部看向控制。

    OmniGibson 内置 head tracking 在目标超出头部 yaw 范围时可能回默认头位；
    这里改成 clamp 到 Tiago 头部关节限位，让头部尽量朝向被选中的物体。
    """
    if not isinstance(robot, Tiago):
        return action

    target_pos = target_pos if isinstance(target_pos, th.Tensor) else th.tensor(target_pos, dtype=th.float32)
    robot_pose = robot.get_position_orientation()
    target_in_base = T.relative_pose_transform(target_pos, th.tensor([0.0, 0.0, 0.0, 1.0]), *robot_pose)[0]

    head1_joint = robot.joints["head_1_joint"]
    head2_joint = robot.joints["head_2_joint"]
    head1_goal = th.atan2(target_in_base[1], target_in_base[0])

    head2_pose = robot.links["head_2_link"].get_position_orientation()
    head2_in_base = T.relative_pose_transform(*head2_pose, *robot_pose)[0]
    horizontal_dist = th.clamp(th.norm(target_in_base[:2]), min=1e-4)
    head2_goal = th.atan2(target_in_base[2] - head2_in_base[2], horizontal_dist)

    head_q = th.stack(
        (
            th.clamp(head1_goal, head1_joint.lower_limit, head1_joint.upper_limit),
            th.clamp(head2_goal, head2_joint.lower_limit, head2_joint.upper_limit),
        )
    )
    action[robot.controller_action_idx["camera"]] = head_q
    return action


def overwrite_head_action_clamped(robot, action, target_obj):
    return overwrite_head_action_clamped_to_position(robot, action, target_obj.get_position_orientation()[0])


def apply_demo_head_tracking(robot, controller, action):
    """
    统一的头部相机追踪入口。

    注意：StarterSemanticActionPrimitives 自带 head tracking 会在 GRASP / PLACE 的 action 上
    改写 camera controller。为了避免它在抓取规划时把头部复位，本 demo 初始化 controller 时
    关闭内置 head tracking，所有 action 都在主循环里走这里。
    """
    if controller is None:
        return action

    action = lock_auxiliary_arm_action(robot, controller, action)

    tracking_point = getattr(controller, "_tracking_point", None)
    if tracking_point is not None:
        return overwrite_head_action_clamped_to_position(robot, action, tracking_point)

    target_obj = controller._tracking_object
    if target_obj is None or target_obj == robot:
        return action
    return overwrite_head_action_clamped(robot, action, target_obj)


def target_position(target):
    """把“物体”或“世界坐标点”统一转成 3D 世界坐标。"""
    if hasattr(target, "get_position_orientation"):
        return target.get_position_orientation()[0]
    return target if isinstance(target, th.Tensor) else th.tensor(target, dtype=th.float32)


def choose_arm_for_target(robot, target):
    """
    根据目标在 robot base frame 中的左右位置选择顺手的手臂。

    Tiago base frame 里 y > 0 通常在左侧，对应 left arm；y < 0 在右侧，对应 right arm。
    目标接近正前方时保留默认手臂，避免细微横向误差导致左右手频繁切换。
    """
    if not hasattr(robot, "arm_names") or set(robot.arm_names) != {"left", "right"}:
        return robot.default_arm

    target_pos = target_position(target)
    target_in_base = T.relative_pose_transform(
        target_pos,
        th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
        *robot.get_position_orientation(),
    )[0]
    if th.abs(target_in_base[1]) < 0.20:
        return robot.default_arm
    return "left" if target_in_base[1] >= 0 else "right"


def select_controller_arm(controller, target=None, prefer_in_hand=True):
    """
    动态设置本次 primitive 使用的手臂。

    抓取时在底盘朝向目标后按目标左右侧选择；放置时如果已有物体在某只手上，
    优先沿用拿物体的手。
    """
    if prefer_in_hand:
        for arm_name in controller.robot.arm_names:
            if controller.robot._ag_obj_in_hand[arm_name] is not None:
                controller._selected_arm = arm_name
                print(f"[arm] Keeping {arm_name} arm because it is holding the object.")
                return arm_name

    arm_name = choose_arm_for_target(controller.robot, target) if target is not None else controller.robot.default_arm
    controller._selected_arm = arm_name
    label = target.name if hasattr(target, "name") else "target"
    print(f"[arm] Selected {arm_name} arm for {label}.")
    return arm_name


def lock_auxiliary_arm_action(robot, controller, action):
    """把非当前操作手臂的 arm / gripper action 固定到当前关节值，避免另一只手无意义运动。"""
    if controller is None or not hasattr(robot, "arm_names"):
        return action

    current_q = robot.get_joint_positions()
    for arm_name in robot.arm_names:
        if arm_name == controller.arm:
            continue
        action[robot.controller_action_idx[f"arm_{arm_name}"]] = current_q[robot.arm_control_idx[arm_name]]
        action[robot.controller_action_idx[f"gripper_{arm_name}"]] = current_q[robot.gripper_control_idx[arm_name]]
    return action


def install_dynamic_single_arm_control(controller):
    """
    让 primitive 每次动态选择顺手手臂，同时冻结辅助臂。

    StarterSemanticActionPrimitives 默认 arm 来自 robot.default_arm；CuRobo 规划返回完整关节轨迹，
    可能让另一只手作为冗余关节一起动。这里在 demo 层把 controller.arm 改成当前选择的手，
    并把规划轨迹里的辅助臂关节覆盖成当前关节。
    """
    controller._selected_arm = controller.robot.default_arm
    type(controller).arm = property(lambda self: getattr(self, "_selected_arm", self.robot.default_arm))

    original_plan_joint_motion = controller._plan_joint_motion

    def plan_joint_motion_lock_auxiliary(self, *args, **kwargs):
        q_traj = original_plan_joint_motion(*args, **kwargs)
        current_q = self.robot.get_joint_positions()
        for arm_name in self.robot.arm_names:
            if arm_name == self.arm:
                continue
            q_traj[:, self.robot.arm_control_idx[arm_name]] = current_q[self.robot.arm_control_idx[arm_name]]
            q_traj[:, self.robot.gripper_control_idx[arm_name]] = current_q[self.robot.gripper_control_idx[arm_name]]
        return q_traj

    controller._plan_joint_motion = MethodType(plan_joint_motion_lock_auxiliary, controller)
    print("[arm] Dynamic arm selection enabled; auxiliary arm will be locked.")


def install_object_facing_navigation(controller):
    """
    优化抓取前的底盘导航采样。

    starter primitive 默认找到第一个可达 pose 就返回，这个 pose 往往只照顾手臂工作空间，
    不一定照顾头部相机视野。这里改成：先生成一批候选 pose，通过 CuRobo 验证可达，
    再优先选择“目标仍在抓取手臂工作空间内，并且头部相机不用转太多”的候选。
    """
    def sample_pose_near_object_facing_target(
        self,
        obj,
        eef_pose=None,
        plan_with_open_gripper=False,
        sampling_attempts=80,
        skip_obstacle_update=False,
    ):
        distance_lo, distance_hi = 0.65, 1.15
        arm_workspace = self.robot.arm_workspace_range[self.arm]
        avg_arm_workspace_range = th.mean(arm_workspace).item()
        head1_joint = self.robot.joints["head_1_joint"] if isinstance(self.robot, Tiago) else None
        head_yaw_limit = (
            max(abs(head1_joint.lower_limit.item()), abs(head1_joint.upper_limit.item()))
            if head1_joint is not None
            else math.pi
        )

        if eef_pose is None:
            eef_pose, _ = self._sample_grasp_pose(obj)
        target_pose = eef_pose

        obj_rooms = (
            obj.in_rooms if obj.in_rooms else [self.robot.scene._seg_map.get_room_instance_by_point(target_pose[0][:2])]
        )

        if not skip_obstacle_update:
            self._motion_generator.update_obstacles()

        def candidate_target_angle(candidate_pose):
            target_direction = math.atan2(
                target_pose[0][1].item() - candidate_pose[1].item(),
                target_pose[0][0].item() - candidate_pose[0].item(),
            )
            return wrap_angle(th.tensor(target_direction, dtype=th.float32) - candidate_pose[2])

        def arm_workspace_penalty(target_angle):
            arm_lo, arm_hi = arm_workspace
            if arm_lo <= target_angle <= arm_hi:
                return th.abs(target_angle - th.tensor(avg_arm_workspace_range, dtype=th.float32)).item()
            return 1.0 + min(th.abs(target_angle - arm_lo).item(), th.abs(target_angle - arm_hi).item())

        def candidate_score(candidate_pose):
            target_angle = candidate_target_angle(candidate_pose)
            # 抓取可靠性优先：目标最好落在默认抓取手臂的工作空间里。
            # 相机视野其次：在满足手臂工作空间的候选中，再尽量减小头部 yaw。
            arm_penalty = arm_workspace_penalty(target_angle)
            limit_penalty = 10.0 if th.abs(target_angle).item() > head_yaw_limit - 0.1 else 0.0
            current_base = self.robot.get_joint_positions()[self.robot.base_control_idx]
            travel_penalty = 0.05 * th.norm(candidate_pose[:2] - current_base[:2]).item()
            return 4.0 * arm_penalty + 0.25 * th.abs(target_angle).item() + limit_penalty + travel_penalty

        attempt = 0
        best_pose = None
        best_score = float("inf")
        valid_candidate_count = 0
        printed_candidate_count = 0
        while attempt < sampling_attempts:
            candidate_poses = []
            for _ in range(self._curobo_batch_size):
                for _ in range(20):
                    radial_yaw = (th.rand(1) * 2.0 * math.pi - math.pi).item()
                    distance = (th.rand(1) * (distance_hi - distance_lo) + distance_lo).item()
                    candidate_xy = target_pose[0][:2] + distance * th.tensor(
                        [math.cos(radial_yaw), math.sin(radial_yaw)], dtype=th.float32
                    )
                    target_direction = radial_yaw + math.pi

                    # desired_target_angle 是“目标在机器人 base frame 里的期望角度”。
                    # 0 表示正前方，arm workspace center 表示原始 primitive 偏好的手臂工作区。
                    desired_target_angles = (
                        0.0,
                        math.copysign(math.radians(15), avg_arm_workspace_range),
                        avg_arm_workspace_range,
                        arm_workspace[0].item(),
                        arm_workspace[1].item(),
                    )

                    room_ok = self.robot.scene._seg_map.get_room_instance_by_point(candidate_xy) in obj_rooms
                    if room_ok:
                        for desired_target_angle in desired_target_angles:
                            base_yaw = wrap_angle(
                                th.tensor(target_direction - desired_target_angle, dtype=th.float32)
                            )
                            candidate_poses.append(th.stack((candidate_xy[0], candidate_xy[1], base_yaw)))
                        break

            if candidate_poses:
                result = self._validate_poses(
                    candidate_poses,
                    eef_pose=target_pose,
                    plan_with_open_gripper=plan_with_open_gripper,
                    skip_obstacle_update=True,
                )
                for i, res in enumerate(result):
                    if res:
                        pose = candidate_poses[i]
                        valid_candidate_count += 1
                        target_angle = candidate_target_angle(pose)
                        arm_center_error = wrap_angle(
                            target_angle - th.tensor(avg_arm_workspace_range, dtype=th.float32)
                        )
                        camera_center_error = target_angle
                        score = candidate_score(pose)
                        if PRINT_NAV_CANDIDATE_DEBUG and printed_candidate_count < MAX_NAV_CANDIDATE_DEBUG_LINES:
                            printed_candidate_count += 1
                            print(
                                "[nav-candidate] "
                                f"pose={[round(v, 3) for v in pose.tolist()]} "
                                f"target_angle={target_angle.item():.3f} "
                                f"arm_expected={avg_arm_workspace_range:.3f} "
                                f"arm_error={arm_center_error.item():.3f} "
                                f"camera_error={camera_center_error.item():.3f} "
                                f"score={score:.3f}"
                            )
                        if score < best_score:
                            best_pose = pose
                            best_score = score

            if (
                best_pose is not None
                and valid_candidate_count >= NAV_MIN_VALID_CANDIDATES_FOR_EARLY_STOP
                and best_score <= NAV_GOOD_SCORE_THRESHOLD
            ):
                print(
                    "[nav] Early stop pose sampling: "
                    f"best score {best_score:.3f} <= {NAV_GOOD_SCORE_THRESHOLD:.3f} "
                    f"after {valid_candidate_count} valid candidates"
                )
                break

            attempt += self._curobo_batch_size

        if best_pose is not None:
            target_angle = candidate_target_angle(best_pose).item()
            arm_error = wrap_angle(
                th.tensor(target_angle, dtype=th.float32) - th.tensor(avg_arm_workspace_range, dtype=th.float32)
            ).item()
            print(
                "[nav] Found grasp-and-camera-friendly base pose: "
                f"{[round(v, 3) for v in best_pose.tolist()]} "
                f"(valid candidates {valid_candidate_count}, "
                f"target angle/head yaw {target_angle:.3f}, "
                f"arm expected {avg_arm_workspace_range:.3f}, "
                f"arm error {arm_error:.3f}, "
                f"score {best_score:.3f})"
            )
        return best_pose

    def navigate_to_pose_with_timing(self, pose_2d, skip_obstacle_update=False):
        pose_list = pose_2d.tolist() if isinstance(pose_2d, th.Tensor) else list(pose_2d)
        print(f"[nav-exec] Start base navigation to {[round(v, 3) for v in pose_list]}")
        started_at = time.perf_counter()
        yielded_steps = 0

        pose_3d = self._get_robot_pose_from_2d_pose(pose_2d)
        if self.debug_visual_marker is not None:
            self.debug_visual_marker.set_position_orientation(*pose_3d)
        q_traj = self._plan_joint_motion(
            target_pos={self.robot.base_footprint_link_name: pose_3d[0]},
            target_quat={self.robot.base_footprint_link_name: pose_3d[1]},
            embodiment_selection=CuRoboEmbodimentSelection.BASE,
            skip_obstacle_update=skip_obstacle_update,
        )

        for action in self._execute_motion_plan(q_traj, low_precision=True):
            yielded_steps += 1
            yield action
        print(f"[nav-exec] Done base navigation: {yielded_steps} action steps in {time.perf_counter() - started_at:.1f}s")

    controller._sample_pose_near_object = MethodType(sample_pose_near_object_facing_target, controller)
    controller._navigate_to_pose = MethodType(navigate_to_pose_with_timing, controller)


def standby_action_generator(robot, controller, steps=STANDBY_STEPS):
    """
    动作执行完后的待机生成器。

    目标：
      1. 手臂/躯干尽量回到 reset_joint_pos；
      2. 底盘不回出生点，保持当前位置；
      3. 如果手里有物体，夹爪保持当前闭合状态；
      4. 每一步都走 apply_demo_head_tracking，让 Tiago 头部相机继续看 tracking object。
    """
    target_q = robot.reset_joint_pos.clone()
    current_q = robot.get_joint_positions()

    # Do not drive the mobile base back to the original spawn pose.
    if hasattr(robot, "base_idx"):
        target_q[robot.base_idx] = current_q[robot.base_idx]

    # Keep the auxiliary arm completely still. Only the selected primitive arm
    # should move during this demo's grasp / place / standby cycle.
    for arm_name in getattr(robot, "arm_names", []):
        if arm_name == controller.arm:
            continue
        target_q[robot.arm_control_idx[arm_name]] = current_q[robot.arm_control_idx[arm_name]]
        target_q[robot.gripper_control_idx[arm_name]] = current_q[robot.gripper_control_idx[arm_name]]

    # If an assisted/sticky grasp is active, keep the gripper closed around the object.
    obj_in_hand = robot._ag_obj_in_hand[controller.arm]
    if obj_in_hand is not None:
        target_q[robot.gripper_control_idx[controller.arm]] = current_q[robot.gripper_control_idx[controller.arm]]

    action = robot.q_to_action(target_q)
    for _ in range(steps):
        yield apply_demo_head_tracking(robot, controller, action)


def look_at_object_generator(robot, controller, obj, steps=LOOK_AT_STEPS):
    """
    抓取前的短暂看向目标动作。

    StarterSemanticActionPrimitives 的 GRASP 本身会在内部设置 _tracking_object，
    但 GRASP 第一次 yield action 之前会先做抓取采样 / CuRobo 规划，这段时间仿真不 step，
    头部就不会马上转过去。这里先让仿真运行几十帧，只保持身体不动并更新头部 action。
    """
    controller._tracking_object = obj
    controller._tracking_point = None
    printed_goal = False
    for _ in range(steps):
        action = robot.q_to_action(robot.get_joint_positions())
        action = overwrite_head_action_clamped(robot, action, obj)
        if not printed_goal:
            printed_goal = True
            if isinstance(robot, Tiago):
                head_action = action[robot.controller_action_idx["camera"]]
                print(f"[look_at] head target for {obj.name}: {[round(v, 3) for v in head_action.tolist()]}")
            else:
                print(f"[look_at] head tracking is only implemented for Tiago, got {type(robot).__name__}")
        yield action


def face_target_generator(robot, controller, target, steps=FACE_TARGET_STEPS):
    """
    在抓取 / 放置规划开始前，先让移动底盘原地朝向目标。

    目标可以是 DatasetObject，也可以是 viewport raycast 得到的世界坐标点。
    底盘先转正后，头部 yaw 需要补偿的角度会更小，更不容易碰到 Tiago 头部限位。
    """
    is_object_target = hasattr(target, "get_position_orientation")
    if is_object_target:
        controller._tracking_object = target
        controller._tracking_point = None
        label = target.name
    else:
        controller._tracking_point = target_position(target).clone()
        label = "clicked point"

    printed_goal = False
    for _ in range(steps):
        current_q = robot.get_joint_positions()
        current_base_q = current_q[robot.base_control_idx]
        target_pos = target_position(target)
        target_xy = target_pos[:2]
        base_xy = current_base_q[:2]

        if th.norm(target_xy - base_xy) < 1e-4:
            break

        target_yaw = th.atan2(target_xy[1] - base_xy[1], target_xy[0] - base_xy[0])
        yaw_error = wrap_angle(target_yaw - current_base_q[2])
        if not printed_goal:
            printed_goal = True
            print(f"[face] base yaw target for {label}: {target_yaw.item():.3f}")
        if th.abs(yaw_error) < 0.04:
            break

        yaw_step = th.clamp(
            yaw_error,
            min=-FACE_TARGET_MAX_YAW_STEP,
            max=FACE_TARGET_MAX_YAW_STEP,
        )
        next_yaw = wrap_angle(current_base_q[2] + yaw_step)
        target_q = current_q.clone()
        target_q[robot.base_control_idx] = th.stack((current_base_q[0], current_base_q[1], next_yaw))
        yield apply_demo_head_tracking(robot, controller, robot.q_to_action(target_q))


def select_arm_generator(robot, controller, target, prefer_in_hand=True):
    """在底盘面向目标后再选择操作手臂，让选手依据最新 base frame。"""
    select_controller_arm(controller, target, prefer_in_hand=prefer_in_hand)
    yield hold_current_pose_action(robot, controller)


def apply_ref_with_debug(controller, primitive, *args, attempts=PRIMITIVE_ATTEMPTS):
    """
    StarterSemanticActionPrimitives.apply_ref 的调试版。

    原版默认会静默重试 5 次；当 GRASP 规划/执行失败时，终端只会看到重复的
    Opening / Sampling / Navigating，很像卡住。这里把每次失败原因打印出来，
    并减少重试次数，方便判断耗时发生在哪里。
    """
    ctrl = controller.controller_functions[primitive]
    errors = []

    for attempt_idx in range(attempts):
        attempt_started_at = time.perf_counter()
        success = False
        try:
            print(f"[attempt] {primitive.name} {attempt_idx + 1}/{attempts}")
            yield from ctrl(*args)
            success = True
        except ActionPrimitiveError as exc:
            errors.append(exc)
            print(f"[attempt-error] {primitive.name} {attempt_idx + 1}/{attempts}: {exc}")

        try:
            if not controller._get_obj_in_hand():
                yield from controller._execute_release()
        except ActionPrimitiveError as exc:
            print(f"[cleanup-warn] release after {primitive.name}: {exc}")

        try:
            yield from controller._reset_robot()
        except ActionPrimitiveError as exc:
            print(f"[cleanup-warn] reset after {primitive.name}: {exc}")

        try:
            yield from controller._settle_robot()
        except ActionPrimitiveError as exc:
            print(f"[cleanup-warn] settle after {primitive.name}: {exc}")

        elapsed = time.perf_counter() - attempt_started_at
        print(f"[attempt-done] {primitive.name} {attempt_idx + 1}/{attempts} in {elapsed:.1f}s")

        if success:
            return

    raise ActionPrimitiveErrorGroup(errors)


def scene_dir(scene_model):
    return os.path.join(DATASET_SCENES_DIR, scene_model)


def scene_json_path(scene_model, scene_file=None):
    if scene_file is not None:
        return os.path.abspath(scene_file)
    return os.path.join(scene_dir(scene_model), "json", f"{scene_model}_best.json")


def task_config_path(scene_model, scene_file=None, task_config_file=None):
    if task_config_file is not None:
        return os.path.abspath(task_config_file)
    return os.path.join(os.path.dirname(scene_json_path(scene_model, scene_file)), "task_config.json")


def load_task_config(scene_model, scene_file=None, task_config_file=None):
    path = task_config_path(scene_model, scene_file, task_config_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"task_config.json not found: {path}")

    with open(path, "r") as f:
        config = json.load(f)

    objects = list(dict.fromkeys(config.get("objects", [])))
    assets = list(dict.fromkeys(config.get("assets", [])))
    if not objects:
        raise ValueError(f"No movable objects listed in task config: {path}")
    if not assets:
        raise ValueError(f"No fixed assets listed in task config: {path}")

    print(f"[task-config] Loaded: {path}")
    print(f"[task-config] objects: {objects}")
    print(f"[task-config] assets: {assets}")
    return {"objects": objects, "assets": assets, "path": path}


def object_category_model(obj):
    return getattr(obj, "category", None), getattr(obj, "model", None)


def print_traversability_map_info(scene):
    trav_map = scene.trav_map
    map_name = "floor_trav_{}.png" if trav_map.trav_map_with_objects else "floor_trav_no_obj_{}.png"
    layout_dir = os.path.join(scene.scene_dir, "layout")
    print("[trav-map] scene_model:", scene.scene_model)
    print("[trav-map] scene_dir:", scene.scene_dir)
    print("[trav-map] layout_dir:", layout_dir)
    print("[trav-map] trav_map_with_objects:", trav_map.trav_map_with_objects)
    for floor in range(trav_map.n_floors):
        print("[trav-map] floor", floor, "file:", os.path.join(layout_dir, map_name.format(floor)))


def build_config(scene_model=DEFAULT_SCENE_MODEL, scene_file=None):
    # 从 tiago_primitives.yaml 继承机器人和控制器配置，再覆盖当前 demo 的场景。
    # 物体不再硬编码创建，而是从 create_scene.py 保存出的 scene JSON 里加载。
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["scene_model"] = scene_model
    config["scene"]["scene_file"] = scene_json_path(scene_model, scene_file)
    config["scene"]["load_object_categories"] = None
    config["objects"] = []
    # Only change Tiago's onboard robot cameras for LeRobot recording.
    # /viewer_camera is still patched separately to skip its RGB annotator.
    config["robots"][0]["sensor_config"] = {
        "VisionSensor": {
            "sensor_kwargs": {
                "image_width": 512,
                "image_height": 512,
            }
        }
    }
    return config


def print_controls(scene_model, task_config):
    print("")
    print("=" * 72)
    print("Interactive mouse pick/place demo")
    print(f"  Scene: {scene_model}")
    print("  Mouse select an Object from task_config: grasp it")
    print("  Mouse click an Asset / surface while holding something: place at clicked point")
    print("")
    print("Keyboard fallback:")
    for idx, obj_name in enumerate(task_config["objects"][:9], start=1):
        print(f"  {idx}: grasp {obj_name}")
    print(f"  P: place held object on first asset: {task_config['assets'][0]}")
    print("  R: release held object at current gripper pose")
    print("")
    print("Recording:")
    print("  B: begin recording episode")
    print("  S: stop and save episode")
    print("  Z: discard current episode")
    print("  F: finalize recorder and quit")
    print("  ESC: quit")
    print("")
    print("The simulation keeps stepping while no command is active.")
    print("=" * 72)
    print("")


def main(scene_model=DEFAULT_SCENE_MODEL, scene_file=None, task_config_file=None, record_fps=30, short_exec=False):
    """
    Interactive pick/place prototype.

    The important structure is:
      1. keep the simulator stepping every loop,
      2. enqueue commands from user input,
      3. execute at most one action primitive generator at a time.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    task_config = load_task_config(scene_model, scene_file=scene_file, task_config_file=task_config_file)
    install_viewer_camera_without_rgb_annotator()
    env = og.Environment(configs=build_config(scene_model=scene_model, scene_file=scene_file))
    scene = env.scene
    robot = env.robots[0]
    print_traversability_map_info(scene)

    for _ in range(30):
        og.sim.step()

    og.sim.enable_viewer_camera_teleoperation()

    # 关闭 starter primitive 内置 head tracking，避免 GRASP / PLACE 过程中它把 camera 关节复位。
    # 本 demo 统一在主循环里用 apply_demo_head_tracking() 控制头部相机。
    controller = StarterSemanticActionPrimitives(env, robot, enable_head_tracking=False)
    controller._tracking_point = None
    install_dynamic_single_arm_control(controller)
    install_object_facing_navigation(controller)
    recorder = LeRobotRecorder(robot=robot, fps=record_fps)
    command_queue = deque()
    should_quit = {"value": False}
    exited_cleanly = False

    pickable_names = [name for name in task_config["objects"] if scene.object_registry("name", name) is not None]
    asset_names = [name for name in task_config["assets"] if scene.object_registry("name", name) is not None]
    missing_pickables = sorted(set(task_config["objects"]) - set(pickable_names))
    missing_assets = sorted(set(task_config["assets"]) - set(asset_names))
    if missing_pickables:
        print(f"[warn] task_config object(s) missing from loaded scene: {missing_pickables}")
    if missing_assets:
        print(f"[warn] task_config asset(s) missing from loaded scene: {missing_assets}")
    if not pickable_names:
        raise RuntimeError("No task_config objects were found in the loaded scene.")
    if not asset_names:
        raise RuntimeError("No task_config assets were found in the loaded scene.")

    place_target = scene.object_registry("name", asset_names[0])
    print("[task] Loaded operation objects:")
    for idx, name in enumerate(pickable_names[:9], start=1):
        obj = scene.object_registry("name", name)
        category, model = object_category_model(obj)
        print(f"  {idx}: {name} ({category} / {model})")
    print("[task] Loaded fixed assets:")
    for name in asset_names:
        obj = scene.object_registry("name", name)
        category, model = object_category_model(obj)
        print(f"  - {name} ({category} / {model})")
    last_selected_prim_path = {"value": None}
    last_mouse_position = {"value": None}
    pending_mouse_click = {"value": False}
    printed_mouse_event_debug = {"value": False}
    printed_no_mouse_xy = {"value": False}
    mouse_callback_id = {"value": None}
    viewport_click_picker = {
        "installed": False,
        "scene_registration": None,
        "manipulator": None,
        "transform": None,
        "screen": None,
        "gesture": None,
        "click_count": 0,
        "query_count": 0,
    }

    def enqueue_grasp(obj_name):
        # 输入事件不直接执行动作，只把“抓取某物体”放进队列。
        # 主循环会在当前 primitive 结束后取出队列并执行。
        obj = scene.object_registry("name", obj_name)
        if obj is None:
            print(f"[warn] Object not found: {obj_name}")
            return
        # 先让 Tiago 头部相机看向被选中的目标，再启动耗时的抓取规划。
        # 这样不会出现“已经开始规划了，但头还没朝向目标”的视觉延迟。
        controller._tracking_object = obj
        controller._tracking_point = None
        command_queue.append((face_target_generator, (robot, controller, obj), obj.name))
        command_queue.append((select_arm_generator, (robot, controller, obj, False), obj.name))
        command_queue.append((look_at_object_generator, (robot, controller, obj), obj.name))
        command_queue.append((StarterSemanticActionPrimitiveSet.GRASP, (obj,), obj.name))
        print(f"[queued] FACE_TARGET + SELECT_ARM + LOOK_AT + GRASP {obj.name}")

    def enqueue_place(target_obj=place_target):
        # 普通 PLACE_ON_TOP：让 starter primitive 自己在目标物体上采样一个可放置点。
        # 这是键盘 P 的兜底逻辑，不保证放到鼠标点击的精确位置。
        controller._tracking_object = target_obj
        controller._tracking_point = None
        command_queue.append((face_target_generator, (robot, controller, target_obj), target_obj.name))
        command_queue.append((select_arm_generator, (robot, controller, target_obj, True), target_obj.name))
        command_queue.append((StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, (target_obj,), target_obj.name))
        print(f"[queued] FACE_TARGET + SELECT_ARM + PLACE_ON_TOP {target_obj.name}")

    def enqueue_place_at(target_obj=place_target, hit_point=None):
        # 精确放置：使用 viewport raycast 得到的 hit_point 作为目标坐标。
        # 如果没有 hit_point，说明当前只知道“选中了哪个物体”，不知道“点在物体哪里”。
        if hit_point is None:
            print("[warn] Precise click point is unavailable; not placing. Use P for sampled PLACE_ON_TOP fallback.")
            return

        hit_point = hit_point.clone() if isinstance(hit_point, th.Tensor) else th.tensor(hit_point, dtype=th.float32)
        controller._tracking_object = target_obj
        controller._tracking_point = hit_point
        command_queue.append((face_target_generator, (robot, controller, hit_point), target_obj.name))
        command_queue.append((select_arm_generator, (robot, controller, hit_point, True), target_obj.name))
        command_queue.append((place_at_point_generator, (target_obj, hit_point), target_obj.name))
        point = hit_point.tolist()
        print(f"[queued] PLACE_AT_POINT {target_obj.name} @ {[round(v, 3) for v in point]}")

    def enqueue_release():
        # 原地释放：只打开夹爪，不规划到新的放置点。
        command_queue.append((StarterSemanticActionPrimitiveSet.RELEASE, tuple(), "current gripper pose"))
        print("[queued] RELEASE")

    def object_from_selected_prim_path(prim_path):
        # Viewport 点击通常返回子 mesh/link 的 prim path，不一定是 OmniGibson object 根节点。
        # 所以一路向父路径查找，直到在 scene registry 里找到对应 object。
        path = str(prim_path)
        while path:
            obj = scene.object_registry("prim_path", path)
            if obj is not None:
                return obj
            path = path.rsplit("/", 1)[0] if "/" in path else ""
        return None

    def object_top_center(obj):
        aabb_min, aabb_max = obj.aabb
        point = (aabb_min + aabb_max) / 2.0
        point[2] = aabb_max[2]
        return point

    def install_viewport_click_picker():
        # 精确点击坐标的主路径：
        # 1. 注册一个 viewport scene manipulator；
        # 2. 鼠标点击时拿到 NDC 坐标；
        # 3. 通过 perform_raycast_query 得到 prim_path 和 world_space_pos；
        # 4. world_space_pos 就是“点在物体表面的世界坐标”。
        try:
            from omni.ui import scene as sc
            from omni.kit.viewport.registry import RegisterScene
            from omni.kit.viewport.window.raycast import perform_raycast_query
        except Exception as exc:
            print(f"[warn] Viewport click picker is unavailable: {exc}")
            return False

        try:
            viewport_api = og.sim.viewer_camera._viewport.viewport_api
        except Exception as exc:
            print(f"[warn] Could not access viewer viewport API: {exc}")
            return False

        class DoNotPrevent(sc.GestureManager):
            def can_be_prevented(self, gesture):
                return False

        def query_completed(prim_path, world_space_pos, *args):
            # raycast 异步返回后进入这里。这里才真正知道点中了哪个 prim、世界坐标是多少。
            if not prim_path:
                print("[select] Viewport click did not hit any prim")
                return

            obj = object_from_selected_prim_path(prim_path)
            if obj is None:
                print(f"[select] No OmniGibson object found for prim: {prim_path}")
                return

            viewport_click_picker["query_count"] += 1
            point = th.tensor(world_space_pos, dtype=th.float32)
            print(f"[select] Viewport hit {obj.name} @ {[round(v, 3) for v in point.tolist()]}")
            handle_selected_object(obj, point)

        class ClickPickGesture(sc.ClickGesture):
            def __init__(self):
                super().__init__(mouse_button=0, manager=DoNotPrevent())

            def on_ended(self, *args):
                # 点击结束时，把 viewport NDC 鼠标坐标映射到渲染纹理像素，再发起 raycast query。
                if self.state == sc.GestureState.CANCELED:
                    return

                viewport_click_picker["click_count"] += 1
                ndc_mouse = self.sender.gesture_payload.mouse
                mouse_pixel, mapped_viewport_api = viewport_api.map_ndc_to_texture_pixel(ndc_mouse)
                if not mouse_pixel or not mapped_viewport_api:
                    print("[select] Viewport click was outside the rendered texture")
                    return

                perform_raycast_query(
                    viewport_api=mapped_viewport_api,
                    mouse_ndc=ndc_mouse,
                    mouse_pixel=mouse_pixel,
                    on_complete_fn=query_completed,
                    query_name="solve_specific_task_click_pick",
                )

        class ClickPickManipulator(sc.Manipulator):
            def __init__(self, viewport_desc):
                super().__init__()
                self.viewport_api = viewport_desc.get("viewport_api", viewport_api)
                self.transform = None
                self.screen = None
                self.gesture = None
                self.name = "SolveSpecificTaskClickPicker"
                self.categories = ("manipulator",)

            def on_build(self):
                self.gesture = ClickPickGesture()
                self.transform = sc.Transform()
                with self.transform:
                    self.screen = sc.Screen(gesture=self.gesture)
                viewport_click_picker["manipulator"] = self
                viewport_click_picker["transform"] = self.transform
                viewport_click_picker["screen"] = self.screen
                viewport_click_picker["gesture"] = self.gesture

            def destroy(self):
                self.screen = None
                self.gesture = None
                if self.transform is not None:
                    self.transform.clear()
                    self.transform = None

        try:
            scene_registration = RegisterScene(
                ClickPickManipulator,
                "omnigibson.examples.action_primitives.solve_specific_task.ClickPicker",
            )
        except Exception as exc:
            print(f"[warn] Could not install viewport click picker: {exc}")
            return False

        viewport_click_picker["installed"] = True
        viewport_click_picker["scene_registration"] = scene_registration
        print("[info] Viewport click picker installed; using precise raycast hit points.")
        return True

    def install_mouse_tracker():
        # 备用路径：如果精确 viewport picker 装不上，尝试监听全局 mouse event。
        # 某些 Isaac / Kit 版本这里拿不到 viewport 内鼠标事件，所以只作为 fallback。
        try:
            appwindow = lazy.omni.appwindow.get_default_app_window()
            input_interface = lazy.carb.input.acquire_input_interface()
            mouse = appwindow.get_mouse()
        except Exception as exc:
            print(f"[warn] Mouse position tracking is unavailable: {exc}")
            return

        def mouse_callback(event, *args, **kwargs):
            xy = extract_mouse_xy(event)
            if xy is not None:
                last_mouse_position["value"] = xy
                if is_mouse_click(event):
                    pending_mouse_click["value"] = True
            elif not printed_mouse_event_debug["value"]:
                printed_mouse_event_debug["value"] = True
                attrs = [name for name in dir(event) if not name.startswith("_")]
                print(f"[debug] Mouse event has no known xy fields. type={getattr(event, 'type', None)}")
                print(f"[debug] Mouse event attrs: {attrs}")
            return True

        try:
            mouse_callback_id["value"] = input_interface.subscribe_to_mouse_events(mouse, mouse_callback)
        except Exception as exc:
            print(f"[warn] Could not subscribe to mouse events: {exc}")

    def extract_mouse_xy(event):
        # Different Isaac / Kit builds expose mouse coordinates under different shapes.
        for attr_name in ("position", "pos", "pixel", "screen_position", "mouse_position"):
            if hasattr(event, attr_name):
                value = getattr(event, attr_name)
                try:
                    return float(value[0]), float(value[1])
                except Exception:
                    if hasattr(value, "x") and hasattr(value, "y"):
                        return float(value.x), float(value.y)

        for x_name, y_name in (
            ("mouse_x", "mouse_y"),
            ("x", "y"),
            ("X", "Y"),
            ("screen_x", "screen_y"),
            ("pixel_x", "pixel_y"),
            ("normalized_x", "normalized_y"),
            ("pos_x", "pos_y"),
            ("position_x", "position_y"),
        ):
            if hasattr(event, x_name) and hasattr(event, y_name):
                return float(getattr(event, x_name)), float(getattr(event, y_name))

        payload = getattr(event, "payload", None)
        if isinstance(payload, dict):
            for x_name, y_name in (
                ("mouse_x", "mouse_y"),
                ("x", "y"),
                ("screen_x", "screen_y"),
                ("pixel_x", "pixel_y"),
                ("normalized_x", "normalized_y"),
            ):
                if x_name in payload and y_name in payload:
                    return float(payload[x_name]), float(payload[y_name])

        return None

    def is_mouse_click(event):
        event_type = str(getattr(event, "type", "")).lower()
        event_input = str(getattr(event, "input", "")).lower()
        return "press" in event_type or "release" in event_type or "left" in event_input

    def clicked_world_point_for_object(obj):
        # fallback 坐标计算。理想情况下不会走到这里，因为精确路径会直接传入 hit_point。
        # 如果走到这里，说明当前只通过 USD selection 知道 obj，但没有精确点击坐标。
        if viewport_click_picker["installed"] and viewport_click_picker["click_count"] == 0:
            print("[warn] Precise viewport click picker is installed but has not received a click event.")
        elif viewport_click_picker["installed"] and viewport_click_picker["query_count"] == 0:
            print("[warn] Precise viewport click picker received clicks, but raycast query has not returned a hit yet.")

        mouse_xy = last_mouse_position["value"]
        if mouse_xy is None:
            if not printed_no_mouse_xy["value"]:
                printed_no_mouse_xy["value"] = True
                print("[warn] No precise mouse hit point has been received yet; ignoring placement click.")
            return None

        try:
            cam = og.sim.viewer_camera
            width, height = cam.image_width, cam.image_height
            x, y = mouse_xy
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                x, y = x * width, y * height
            x = max(0.0, min(float(width - 1), x))
            y = max(0.0, min(float(height - 1), y))

            cam_pos, cam_quat = cam.get_position_orientation()
            rot = T.quat2mat(cam_quat)
            K = cam.intrinsic_matrix
            x_cam = ((x - K[0, 2]) / K[0, 0]).item()
            y_cam = ((y - K[1, 2]) / K[1, 1]).item()
            local_dirs = (
                th.tensor([x_cam, -y_cam, -1.0], dtype=th.float32),
                th.tensor([x_cam, y_cam, 1.0], dtype=th.float32),
            )

            fallback_hit = None
            for local_dir in local_dirs:
                world_dir = rot @ (local_dir / th.norm(local_dir))
                result = og.sim.psqi.raycast_closest(
                    origin=cam_pos.tolist(),
                    dir=world_dir.tolist(),
                    distance=100.0,
                )
                if not result["hit"]:
                    continue

                hit_obj = object_from_selected_prim_path(result.get("rigidBody", result.get("collision", "")))
                if hit_obj == obj:
                    return th.tensor(result["position"], dtype=th.float32)
                if fallback_hit is None:
                    fallback_hit = th.tensor(result["position"], dtype=th.float32)

            if fallback_hit is not None:
                return fallback_hit
        except Exception as exc:
            print(f"[warn] Could not compute clicked world point: {exc}")

        return None

    def place_at_point_generator(target_obj, hit_point):
        # 自定义“精确放置”动作生成器。
        # starter primitive 的 PLACE_ON_TOP 会自己采样放置点；这里我们改成使用鼠标点击的 hit_point。
        obj_in_hand = robot._ag_obj_in_hand[controller.arm]
        if obj_in_hand is None:
            raise RuntimeError("You need to be grasping an object before placing at a clicked point.")

        # 放置时让 Tiago 头部看向目标支撑物体。
        controller._tracking_object = target_obj
        hit_point = hit_point.clone() if isinstance(hit_point, th.Tensor) else th.tensor(hit_point, dtype=th.float32)
        controller._tracking_point = hit_point

        # 目标是“物体底部落在 hit_point 上”，所以需要根据当前 AABB 计算 root 到底部的高度偏移。
        obj_pos, obj_quat = obj_in_hand.get_position_orientation()
        obj_aabb_min, _ = obj_in_hand.aabb
        bottom_to_root_z = obj_pos[2] - obj_aabb_min[2]
        desired_obj_pos = obj_pos.clone()
        desired_obj_pos[:2] = hit_point[:2]
        desired_obj_pos[2] = hit_point[2] + bottom_to_root_z + 0.01

        robot_pos = robot.get_position_orientation()[0]
        if th.norm(desired_obj_pos[:2] - robot_pos[:2]) > MAX_CLICK_PLACE_DISTANCE:
            raise RuntimeError(
                f"Clicked point is too far from the robot for this demo "
                f"({th.norm(desired_obj_pos[:2] - robot_pos[:2]).item():.2f}m)."
            )

        hand_pose = controller._get_hand_pose_for_object_pose((desired_obj_pos, obj_quat))

        # 先判断机械臂当前位置能不能直接够到；够不到再尝试导航到底盘可达位置。
        controller._motion_generator.update_obstacles()
        initial_joint_pos = controller._get_joint_position_with_fingers_at_limit("upper")
        target_in_reach = controller._target_in_reach_of_robot(
            hand_pose,
            initial_joint_pos=initial_joint_pos,
            skip_obstacle_update=True,
        )

        if target_in_reach:
            yield from controller._move_hand(hand_pose)
        else:
            # 根据目标末端位姿，在目标物体附近采样一个底盘导航位姿。
            nav_pose = controller._sample_pose_near_object(
                target_obj,
                eef_pose=hand_pose,
                plan_with_open_gripper=True,
                sampling_attempts=10,
                skip_obstacle_update=True,
            )
            if nav_pose is None:
                print(f"[warn] Could not find a valid navigation pose for clicked point on {target_obj.name}.")
                return
            yield from controller._navigate_to_pose(nav_pose, skip_obstacle_update=True)
            yield from controller._move_hand(hand_pose)

        # 到达放置位姿后打开夹爪，完成释放。
        yield from controller._execute_release()

    def handle_selected_object(obj, hit_point=None):
        # 鼠标点击/键盘命令最终都汇聚到这里：
        #   - 手里没东西：如果点的是可抓物体，就入队 GRASP；
        #   - 手里有东西：如果点的是其他物体/位置，就入队 PLACE_AT_POINT。
        if active_generator is not None:
            print(f"[select] Ignoring {obj.name}; currently executing {active_command}")
            return
        if command_queue:
            print(f"[select] Ignoring {obj.name}; command queue is not empty")
            return

        obj_in_hand = robot._ag_obj_in_hand[controller.arm]
        if obj_in_hand is None:
            if obj.name in pickable_names:
                enqueue_grasp(obj.name)
            else:
                print(f"[select] {obj.name} is not pickable. Select one of: {pickable_names}")
        else:
            if obj == obj_in_hand:
                print(f"[select] {obj.name} is already in hand")
            elif obj.name not in asset_names:
                print(f"[select] {obj.name} is not marked as an asset/place target. Assets: {asset_names}")
            else:
                enqueue_place_at(obj, hit_point if hit_point is not None else clicked_world_point_for_object(obj))

    def poll_viewport_selection():
        # fallback 输入路径：轮询 USD 当前选中的 prim。
        # 这个路径只能知道选中了哪个 object，不能稳定知道精确点击坐标。
        selection = lazy.omni.usd.get_context().get_selection().get_selected_prim_paths()
        if not selection:
            return

        prim_path = selection[-1]
        if prim_path == last_selected_prim_path["value"] and not pending_mouse_click["value"]:
            return 
        last_selected_prim_path["value"] = prim_path
        pending_mouse_click["value"] = False

        obj = object_from_selected_prim_path(prim_path)
        if obj is None:
            print(f"[select] No OmniGibson object found for prim: {prim_path}")
            return

        handle_selected_object(obj)

    for key, obj_name in enumerate(pickable_names[:9], start=1):
        keyboard_key = getattr(lazy.carb.input.KeyboardInput, f"KEY_{key}")
        KeyboardEventHandler.add_keyboard_callback(
            key=keyboard_key,
            callback_fn=lambda name=obj_name: enqueue_grasp(name),
        )

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.P,
        callback_fn=enqueue_place,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.R,
        callback_fn=enqueue_release,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.B,
        callback_fn=recorder.start_recording,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.S,
        callback_fn=recorder.stop_and_save_episode,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.Z,
        callback_fn=recorder.discard_episode,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.F,
        callback_fn=lambda: should_quit.__setitem__("value", True),
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.ESCAPE,
        callback_fn=lambda: should_quit.__setitem__("value", True),
    )

    active_generator = None
    active_command = None
    active_is_standby = False
    active_started_at = None
    steps = 0
    max_steps = 1000 if short_exec else -1

    if not install_viewport_click_picker():
        install_mouse_tracker()

    print_controls(scene_model, task_config)
    obs, _, _, _, _ = env.step(action=hold_current_pose_action(robot, controller))
    steps += 1

    while steps != max_steps:
        if should_quit["value"]:
            if recorder.finalize_and_exit():
                exited_cleanly = True
                break
            should_quit["value"] = False

        # 每一帧都保持仿真 step。没有命令时机器人保持当前姿态，有命令时执行一个 generator 的下一步。
        if not active_is_standby:
            poll_viewport_selection()

        if active_generator is None and command_queue:
            # 队列中每条命令都是一个 primitive 或自定义 generator。
            # GRASP / PLACE_ON_TOP / RELEASE 走 controller.apply_ref；
            # place_at_point_generator 这种本地函数直接调用。
            primitive, args, label = command_queue.popleft()
            active_command = f"{primitive.name if hasattr(primitive, 'name') else primitive.__name__} {label}"
            active_is_standby = False
            active_started_at = time.perf_counter()
            print(f"[start] {active_command}")
            active_generator = (
                apply_ref_with_debug(controller, primitive, *args)
                if hasattr(primitive, "name")
                else primitive(*args)
            )

        if active_generator is not None:
            try:
                # generator 每 next 一次，产出当前仿真 step 要执行的 action。
                action = next(active_generator)
                if action is None:
                    action = hold_current_pose_action(robot, controller)
                else:
                    action = apply_demo_head_tracking(robot, controller, action)
            except StopIteration:
                # 一个抓取/释放/放置动作结束后，自动切到 STANDBY，让机器人回到待机姿态。
                elapsed = time.perf_counter() - active_started_at if active_started_at is not None else 0.0
                print(f"[done] {active_command} in {elapsed:.1f}s")
                if active_is_standby:
                    active_generator = None
                    active_command = None
                    active_is_standby = False
                    active_started_at = None
                    action = hold_current_pose_action(robot, controller)
                elif active_command is not None and active_command.startswith(
                    ("face_target_generator", "select_arm_generator", "look_at_object_generator")
                ) and command_queue:
                    # FACE_TARGET / LOOK_AT 都是正式 primitive 的前置动作，结束后应立即进入下一条命令。
                    # 如果这里插入 STANDBY，底盘朝向或头部追踪会被待机动作打断。
                    active_generator = None
                    active_command = None
                    active_is_standby = False
                    active_started_at = None
                    action = hold_current_pose_action(robot)
                else:
                    active_generator = standby_action_generator(robot, controller)
                    active_command = "STANDBY"
                    active_is_standby = True
                    active_started_at = time.perf_counter()
                    action = next(active_generator)
            except Exception as exc:
                elapsed = time.perf_counter() - active_started_at if active_started_at is not None else 0.0
                print(f"[error] {active_command} after {elapsed:.1f}s: {exc}")
                active_generator = None
                active_command = None
                active_is_standby = False
                active_started_at = None
                action = hold_current_pose_action(robot, controller)
        else:
            action = hold_current_pose_action(robot, controller)

        state_24d = get_24d_state(robot)
        next_obs, _, _, _, _ = env.step(action)
        recorder.step(obs_dict=next_obs, state=state_24d, action=action)
        if recorder.is_recording and len(recorder.episode_buffer) > 0 and len(recorder.episode_buffer) % record_fps == 0:
            print(f"[record] captured {len(recorder.episode_buffer)} frames", end="\r")
        obs = next_obs
        steps += 1

    if mouse_callback_id["value"] is not None:
        try:
            appwindow = lazy.omni.appwindow.get_default_app_window()
            input_interface = lazy.carb.input.acquire_input_interface()
            input_interface.unsubscribe_to_mouse_events(appwindow.get_mouse(), mouse_callback_id["value"])
        except Exception:
            pass
    if viewport_click_picker["scene_registration"] is not None:
        try:
            viewport_click_picker["scene_registration"].destroy()
        except Exception:
            pass
    if viewport_click_picker["transform"] is not None:
        try:
            viewport_click_picker["transform"].clear()
        except Exception:
            pass
    KeyboardEventHandler.reset()
    if not exited_cleanly:
        recorder.finalize_and_exit()
    og.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-model",
        default=DEFAULT_SCENE_MODEL,
        help="Scene directory name under datasets/behavior-1k-assets/scenes.",
    )
    parser.add_argument(
        "--scene-file",
        default=None,
        help="Optional full path to a scene JSON. Defaults to <scene-model>/json/<scene-model>_best.json.",
    )
    parser.add_argument(
        "--task-config",
        default=None,
        help="Optional full path to task_config.json. Defaults to the JSON directory for the selected scene.",
    )
    parser.add_argument("--record-fps", type=int, default=30, help="LeRobot recording FPS.")
    parser.add_argument("--short-exec", action="store_true", help="Run for a short smoke-test horizon.")
    args = parser.parse_args()
    main(
        scene_model=args.scene_model,
        scene_file=args.scene_file,
        task_config_file=args.task_config,
        record_fps=args.record_fps,
        short_exec=args.short_exec,
    )

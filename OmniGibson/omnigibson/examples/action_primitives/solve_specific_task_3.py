import math
import os
import time
from collections import deque
from types import MethodType

import torch as th
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, ActionPrimitiveErrorGroup
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.robots.tiago import Tiago
from omnigibson.utils.geometry_utils import wrap_angle
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import KeyboardEventHandler


# 可被抓取的物体配置。字典 key 同时作为键盘快捷键，例如按 3 抓 lemon sauce。
PICKABLE_OBJECTS = {
    "1": {
        "type": "DatasetObject",
        "name": "cold_cream",
        "category": "bottle_of_cold_cream",
        "model": "lyzvuk",
        "position": [-0.30, -0.80, 0.55],
        "orientation": [0, 0, 0, 1],
    },
    "2": {
        "type": "DatasetObject",
        "name": "wine_bottle",
        "category": "bottle_of_wine",
        "model": "bmudli",
        "position": [-0.30, -1.05, 0.55],
        "orientation": [0, 0, 0, 1],
    },
    "3": {
        "type": "DatasetObject",
        "name": "bottle_of_lemon_sauce",
        "category": "bottle_of_lemon_sauce",
        "model": "iyijeb",
        "position": [-0.30, -1.35, 0.55],
        "orientation": [0, 0, 0, 1],
    },
    "4": {
        "type": "DatasetObject",
        "name": "apple_pie",
        "category": "apple_pie",
        "model": "ejrgdj",
        "position": [-0.30, -1.55, 0.55],
        "orientation": [0, 0, 0, 1],
    },
}

PLACE_TARGET_NAME = "place_table"
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


def install_object_facing_navigation(controller):
    """
    优化抓取前的底盘导航采样。

    starter primitive 默认找到第一个可达 pose 就返回，这个 pose 往往只照顾手臂工作空间，
    不一定照顾头部相机视野。这里改成：先生成一批候选 pose，通过 CuRobo 验证可达，
    再优先选择“目标仍在抓取手臂工作空间内，并且头部相机不用转太多”的候选。
    """
    original_navigate_to_pose = controller._navigate_to_pose

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
        for action in original_navigate_to_pose(pose_2d, skip_obstacle_update=skip_obstacle_update):
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


def build_config():
    # 从 tiago_primitives.yaml 继承机器人和控制器配置，再覆盖当前 demo 的场景和物体。
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
    config["objects"] = [
        *PICKABLE_OBJECTS.values(),
        {
            "type": "DatasetObject",
            "name": PLACE_TARGET_NAME,
            "category": "sofa",
            "model": "bskyog",
            "scale": [0.5, 0.5, 0.5],
            "position": [1.70, -1.1, 0.20],
            "orientation": [0, 0, 1, 0],
        },
    ]
    return config


def print_controls():
    print("")
    print("=" * 72)
    print("Interactive mouse pick/place demo")
    print("  Mouse select a pickable object: grasp it")
    print(f"  Mouse click {PLACE_TARGET_NAME}: place held object at the clicked point")
    print("")
    print("Keyboard fallback:")
    for key, obj_cfg in PICKABLE_OBJECTS.items():
        print(f"  {key}: grasp {obj_cfg['name']} ({obj_cfg['category']} / {obj_cfg['model']})")
    print(f"  P: place held object on {PLACE_TARGET_NAME}")
    print("  R: release held object at current gripper pose")
    print("  ESC: quit")
    print("")
    print("The simulation keeps stepping while no command is active.")
    print("=" * 72)
    print("")


def main(short_exec=False):
    """
    Interactive pick/place prototype.

    The important structure is:
      1. keep the simulator stepping every loop,
      2. enqueue commands from user input,
      3. execute at most one action primitive generator at a time.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    env = og.Environment(configs=build_config())
    scene = env.scene
    robot = env.robots[0]

    for _ in range(30):
        og.sim.step()

    og.sim.enable_viewer_camera_teleoperation()

    # 关闭 starter primitive 内置 head tracking，避免 GRASP / PLACE 过程中它把 camera 关节复位。
    # 本 demo 统一在主循环里用 apply_demo_head_tracking() 控制头部相机。
    controller = StarterSemanticActionPrimitives(env, robot, enable_head_tracking=False)
    controller._tracking_point = None
    install_object_facing_navigation(controller)
    command_queue = deque()
    should_quit = {"value": False}

    pickable_names = [cfg["name"] for cfg in PICKABLE_OBJECTS.values()]
    place_target = scene.object_registry("name", PLACE_TARGET_NAME)
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
        command_queue.append((look_at_object_generator, (robot, controller, obj), obj.name))
        command_queue.append((StarterSemanticActionPrimitiveSet.GRASP, (obj,), obj.name))
        print(f"[queued] FACE_TARGET + LOOK_AT + GRASP {obj.name}")

    def enqueue_place(target_obj=place_target):
        # 普通 PLACE_ON_TOP：让 starter primitive 自己在目标物体上采样一个可放置点。
        # 这是键盘 P 的兜底逻辑，不保证放到鼠标点击的精确位置。
        controller._tracking_object = target_obj
        controller._tracking_point = None
        command_queue.append((face_target_generator, (robot, controller, target_obj), target_obj.name))
        command_queue.append((StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, (target_obj,), target_obj.name))
        print(f"[queued] FACE_TARGET + PLACE_ON_TOP {target_obj.name}")

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

    for key, obj_cfg in PICKABLE_OBJECTS.items():
        keyboard_key = getattr(lazy.carb.input.KeyboardInput, f"KEY_{key}")
        KeyboardEventHandler.add_keyboard_callback(
            key=keyboard_key,
            callback_fn=lambda name=obj_cfg["name"]: enqueue_grasp(name),
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

    print_controls()

    while not should_quit["value"] and steps != max_steps:
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
                    ("face_target_generator", "look_at_object_generator")
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

        env.step(action)
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
    og.shutdown()


if __name__ == "__main__":
    main()

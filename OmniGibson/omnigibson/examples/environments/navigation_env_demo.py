import os
import yaml
import numpy as np
import omnigibson as og
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.utils.transform_utils import quat2euler

# ======================================================================
# 独立封装的底层控制器 (Tracker)：不污染主干业务逻辑
# ======================================================================
def calculate_smooth_action(robot_pos, robot_quat, target_xy):
    """
    标准的 P-Controller (比例控制器) 轨迹追踪器
    作用：根据机器人当前姿态和下一个目标点，平滑计算出最佳的油门和转向
    """
    # 1. 计算小车车头与目标的角度差
    current_yaw = quat2euler(robot_quat)[2]
    dy = target_xy[1] - robot_pos[1]
    dx = target_xy[0] - robot_pos[0]
    target_yaw = np.arctan2(dy, dx)
    
    angle_diff = target_yaw - current_yaw
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    
    # 2. 比例控制 (P-Control) 决定动作
    # 转向：偏差越大，打方向盘越狠；偏差小就微微调整 (限制在最大转速 [-1, 1] 内)
    turn_speed = np.clip(angle_diff * 3.0, -1.0, 1.0)
    
    if abs(angle_diff) > 0.3:
        # 如果车头偏离超过约 17 度：踩死刹车，原地打方向盘对准目标
        forward_speed = 0.0
    else:
        # 如果基本对准了：根据对准的精准度踩油门，越准开得越快 (自适应油门)
        forward_speed = np.clip(1.0 - (abs(angle_diff) / 0.3), 0.0, 1.0)
        
    return np.array([forward_speed, turn_speed], dtype=np.float32)

# ======================================================================
# 主程序
# ======================================================================
def main(random_selection=False, headless=False, short_exec=False):
    """
    这是一个使用 OmniGibson 底层 A* 规划器和 P 控制器
    实现的完整机器人自动驾驶/导航 Demo。
    """
    # 修复了报错点：这里用了 str(main.__doc__) 或者给个默认字符串来容错
    desc = main.__doc__ if main.__doc__ else "No description"
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + desc + "\n    " + "*" * 80)

    # 1. 加载配置与环境
    config_filename = os.path.join(og.example_config_path, "turtlebot_nav.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    load_options = {
        "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
        "Full": "Load all interactive objects in the scene",
    }
    load_mode = choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
    if load_mode == "Quick":
        config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    env = og.Environment(configs=config)
    og.sim.enable_viewer_camera_teleoperation()

    max_iterations = 50 if not short_exec else 1
    
    for j in range(max_iterations):
        og.log.info(f"========= 正在开始第 {j+1} 回合 =========")
        env.reset()
        
        for i in range(220):
            # ---------------------------------------------------------
            # 核心步骤 A: 调用官方底层 A* 引擎进行全局路径规划 (Global Planning)
            # ---------------------------------------------------------
            shortest_path, geodesic_dist = env.task.get_shortest_path_to_goal(env, entire_path=True)
            
            if shortest_path is None or len(shortest_path) < 2:
                # 卡死或者已经到达终点，原地停车
                action = np.array([0.0, 0.0], dtype=np.float32)
            else:
                # ---------------------------------------------------------
                # 核心步骤 B: 提取轨迹前方的路点 (Lookahead Waypoint)
                # ---------------------------------------------------------
                target_idx = min(3, len(shortest_path) - 1)
                target_xy = shortest_path[target_idx]  
                print(f"第 {i} 步：目标点为 {target_xy}")
                # ---------------------------------------------------------
                # 核心步骤 C: 调用追踪器执行驾驶动作 (Local Control)
                # ---------------------------------------------------------
                robot_pos, robot_quat = env.robots[0].get_position_orientation()
                action = calculate_smooth_action(robot_pos, robot_quat, target_xy)

            # 将动作发送给物理仿真引擎执行一帧
            state, reward, terminated, truncated, info = env.step(action)

            # 打印监控信息
            if i % 10 == 0:
                print(f"步数:{i:3d} | 真实避障距离: {geodesic_dist:.2f}米 | 油门:{action[0]:.2f} 转向:{action[1]:.2f}")
                
            if terminated or truncated:
                if reward > 5.0:
                    print(f"神级自动驾驶！完美沿着官方规划轨迹到达终点！耗时 {i + 1} 步\n")
                else:
                    print(f"发生异常或超时！\n")
                break

    # 结束仿真
    og.shutdown()

if __name__ == "__main__":
    main()
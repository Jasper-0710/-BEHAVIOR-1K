import os

import yaml
import numpy as np
import omnigibson as og
from omnigibson.utils.ui_utils import choose_from_options


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select a type of scene and loads a turtlebot into it, generating a Point-Goal navigation
    task within the environment.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Load the config
    config_filename = os.path.join(og.example_config_path, "turtlebot_nav.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # check if we want to quick load or full load the scene
    load_options = {
        "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
        "Full": "Load all interactive objects in the scene",
    }
    load_mode = choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
    if load_mode == "Quick":
        config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # Load the environment
    env = og.Environment(configs=config)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Run a simple loop and reset periodically
    max_iterations = 50 if not short_exec else 1
    # for j in range(max_iterations):
    #     og.log.info("Resetting environment")
    #     env.reset()
    #     for i in range(100):
    #         action = env.action_space.sample()
    #         state, reward, terminated, truncated, info = env.step(action)
    #         if i % 2 == 0:
    #             print(f"正在运行第 {j+1} 回合 - 当前步数: {i}/1000 | 获得奖励: {reward:.4f}")
    #         if terminated or truncated:
    #             og.log.info("Episode finished after {} timesteps".format(i + 1))
    #             break
    for j in range(max_iterations):
        og.log.info(f"========= 正在开始第 {j+1} 回合 =========")
        env.reset()
        
        # 新增一个变量，用来记住上一步的奖励
        last_reward = 0.0  
        
        # 开局默认动作：先往前走走看
        action = np.array([1.0, 0.0], dtype=np.float32)
        for i in range(1000):
            # 1. 闭着眼睛把动作发给3引擎，获取打分
            state, reward, terminated, truncated, info = env.step(action)

            # ==========================================================
            # 2. 纯 Reward 驱动的导航大脑 (基于规则的梯度上升)
            # ==========================================================
            if reward < -0.05:
                # 状态 A：扣分极大，说明撞墙了！(通常在 -0.1 左右)
                # 动作：挂倒挡，并向左猛打方向盘逃离
                action = np.array([-0.5, 1.0], dtype=np.float32)
                
            elif reward < 0.0:
                # 状态 B：扣分很小，说明没撞墙，但偏离目标了
                # 动作：不要直线开！一边缓慢往前开，一边向右转圈，寻找有得分的方向（螺旋寻路）
                action = np.array([0.2, -1.0], dtype=np.float32)
                
            elif reward > 0.001:
                # 状态 C：拿到正分数了！说明当前的朝向是对的
                # 动作：立刻锁定方向，停止转弯，全速直线冲锋！
                action = np.array([1.0, 0.0], dtype=np.float32)
                
            else:
                # 状态 D：奖励极小或为 0 (通常是在原地转圈时没有位移)
                # 保持上一帧的动作不变，让子弹飞一会儿
                pass

            # 打印实时的决策过程
            print(f"步数:{i:3d} | 奖励: {reward:7.4f} | 下一步动作: [{action[0]:5.2f}, {action[1]:5.2f}]")
                
            if terminated or truncated:
                if reward > 5.0:
                    print(f"🏆 [第 {j+1} 回合] 奇迹发生！盲走成功到达终点！耗时 {i + 1} 步\n")
                else:
                    print(f"💥 [第 {j+1} 回合] 被困死了或者超时！回合结束\n")
                break

    # Always close the environment at the end
    og.shutdown()


if __name__ == "__main__":
    main()

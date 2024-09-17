import numpy as np
from metadrive import SafeMetaDriveEnv
from metadrive.component.pgblock.first_block import FirstPGBlock
import csv
from pe_rlhf.utils.policy.IDM_MOBIL_env import IDM_MOBILPolicy


EPISODE_NUM = 10
HORIZON = 2000


env_config = {
    "traffic_density": 0.06,
    "use_render": False,
    'environment_num': 10,
    "start_seed": 129,
    "vehicle_config": {
        "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 2),
    }
}

env = SafeMetaDriveEnv(env_config)


with open('idm_mobil_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Success", "Reward", "Cost", "Mean Velocity", "Mean Acceleration", "Overtake Vehicle Num", "Mean Position"])

    total_success = 0
    total_reward = 0
    total_cost = 0
    total_velocity = 0
    total_acceleration = 0
    total_overtake = 0
    total_position = 0

    for episode in range(EPISODE_NUM):
        o = env.reset()
        agent = IDM_MOBILPolicy(env.vehicle, env.current_seed)

        episode_reward = 0
        episode_cost = 0
        episode_step = 0
        velocity_list = []
        acceleration_list = []
        position_list = []
        previous_velocity = 0

        for step in range(HORIZON):
            action = agent.act()
            o, r, d, info = env.step(action)

            episode_reward += r
            episode_cost += info["cost"]
            velocity_list.append(info["velocity"])
            acceleration = info["velocity"] - previous_velocity
            acceleration_list.append(acceleration)
            position_list.append(info.get("current_position", 0))  # Assuming current_position is available in info
            previous_velocity = info["velocity"]

            episode_step += 1

            if d:
                break

        success = 1 if info.get("arrive_dest", False) else 0
        mean_velocity = np.mean(velocity_list)
        mean_acceleration = np.mean(acceleration_list)
        mean_position = np.mean(position_list)
        overtake_num = info.get("overtake_vehicle_num", 0)

        writer.writerow([episode, success, episode_reward, episode_cost, mean_velocity, mean_acceleration, overtake_num, mean_position])

        total_success += success
        total_reward += episode_reward
        total_cost += episode_cost
        total_velocity += mean_velocity
        total_acceleration += mean_acceleration
        total_overtake += overtake_num
        total_position += mean_position

        print(f"Episode {episode}: Success={success}, Reward={episode_reward:.2f}, Cost={episode_cost:.2f}, "
              f"Mean Velocity={mean_velocity:.2f}, Mean Acceleration={mean_acceleration:.2f}, "
              f"Overtake Num={overtake_num}, Mean Position={mean_position:.2f}")


    avg_success_rate = total_success / EPISODE_NUM
    avg_reward = total_reward / EPISODE_NUM
    avg_cost = total_cost / EPISODE_NUM
    avg_velocity = total_velocity / EPISODE_NUM
    avg_acceleration = total_acceleration / EPISODE_NUM
    avg_overtake = total_overtake / EPISODE_NUM
    avg_position = total_position / EPISODE_NUM

    print("\nOverall Performance:")
    print(f"Success Rate: {avg_success_rate:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Cost: {avg_cost:.2f}")
    print(f"Average Velocity: {avg_velocity:.2f}")
    print(f"Average Acceleration: {avg_acceleration:.2f}")
    print(f"Average Overtake Num: {avg_overtake:.2f}")
    print(f"Average Position: {avg_position:.2f}")

env.close()
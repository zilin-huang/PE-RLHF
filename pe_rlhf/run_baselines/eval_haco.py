import os.path

from pe_rlhf.algo.haco.haco import HACOTrainer
from pe_rlhf.utils.human_in_the_loop_env import HumanInTheLoopEnv
from pe_rlhf.utils.train_utils import initialize_ray

import csv


def get_function(exp_path, ckpt_idx):
    ckpt = os.path.join(exp_path, "checkpoint_{}".format(ckpt_idx), "checkpoint-{}".format(ckpt_idx))
    trainer = HACOTrainer(dict(env=HumanInTheLoopEnv))

    trainer.restore(ckpt)

    def _f(obs):
        ret = trainer.compute_actions({"default_policy": obs})
        return ret

    return _f


if __name__ == '__main__':
    # hyperparameters
    CKPT_PATH ='/home/sky-lab/codes/PE-RLHF/pe_rlhf/run_baselines/haco/HACOTrainer_HumanInTheLoopEnv_656a2_00000_0_seed=0_2024-06-01_18-51-04'

    EPISODE_NUM_PER_CKPT = 1
    CKPT_START = 10
    CKPT_END = 11

    RENDER = True
    env_config = {
        "manual_control": True,
        "use_render": True,
        "controller": "keyboard",
        "window_size": (1600, 1100),
        "cos_similarity": True,
        "map": "TCO",
        "environment_num": 1,
        "start_seed": 15,
    }

    initialize_ray(test_mode=False, local_mode=False, num_gpus=1)


    def make_env(env_cfg=None):
        env_cfg = env_cfg or {}
        env_cfg.update(dict(manual_control=False, use_render=RENDER))
        return HumanInTheLoopEnv(env_cfg)


    from collections import defaultdict

    super_data = defaultdict(list)
    super_velocity_data = defaultdict(list)
    super_acceleration_data = defaultdict(list)
    super_position_data = defaultdict(list)  # New data structure for positions

    env = make_env(env_config)

    # Create a CSV file and a writer
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # In your CSV header
        writer.writerow(["CKPT", "Success Rate", "Mean Episode Reward", "Mean Episode Cost",
                         "Mean Velocity", "Mean Acceleration",
                         "Mean Overtake Vehicle Num", "Mean Position"])

        for ckpt_idx in range(CKPT_START, CKPT_END, 10):
            ckpt = ckpt_idx

            compute_actions = get_function(CKPT_PATH, ckpt_idx)

            o = env.reset()
            epi_num = 0

            total_cost = 0
            total_reward = 0
            success_rate = 0
            ep_cost = 0
            ep_reward = 0
            success_flag = False
            horizon = 2000
            step = 0

            total_overtake_vehicle_num = 0
            ep_overtake_vehicle_num = 0

            velocity_list = []  # List to store velocity data
            acceleration_list = []  # List to store acceleration data
            position_list = []  # List to store position data
            previous_velocity = 0  # Placeholder for the velocity in the previous step

            while True:
                step += 1
                action_to_send = compute_actions(o)["default_policy"]
                o, r, d, info = env.step(action_to_send)

                total_reward += r
                ep_reward += r
                total_cost += info["cost"]
                ep_cost += info["cost"]

                ep_overtake_vehicle_num += info["overtake_vehicle_num"]

                # Add the velocity to the velocity_list
                velocity_list.append(info["velocity"])

                # Calculate acceleration (current velocity - previous velocity)
                acceleration = info["velocity"] - previous_velocity
                acceleration_list.append(acceleration)

                # Add the position to the position_list
                position_list.append(info["current_position"])

                # Save current velocity for the next step
                previous_velocity = info["velocity"]

                if d or step > horizon:
                    if info["arrive_dest"]:
                        success_rate += 1
                        success_flag = True
                    epi_num += 1

                    total_overtake_vehicle_num += ep_overtake_vehicle_num

                    # Inside the episode loop
                    avg_velocity = sum(velocity_list) / step if step > 0 else 0
                    avg_acceleration = sum(acceleration_list) / step if step > 0 else 0
                    avg_position = sum(position_list) / step if step > 0 else 0

                    super_data[ckpt].append({
                        "reward": ep_reward,
                        "success": success_flag,
                        "cost": ep_cost,
                        "overtake_vehicle_num": ep_overtake_vehicle_num,
                        "avg_velocity": avg_velocity,
                        "avg_acceleration": avg_acceleration,
                        "avg_position": avg_position
                    })

                    super_velocity_data[ckpt].append({"velocity": velocity_list * 10})
                    super_acceleration_data[ckpt].append({"acceleration": acceleration_list * 10})
                    super_position_data[ckpt].append({"position": position_list * 10})

                    ep_cost = 0.0
                    ep_reward = 0.0
                    ep_overtake_vehicle_num = 0
                    success_flag = False
                    step = 0

                    velocity_list = []  # Reset velocity list for next episode
                    acceleration_list = []  # Reset acceleration list for next episode
                    position_list = []  # Reset position list for next episode

                    if epi_num >= EPISODE_NUM_PER_CKPT:
                        break
                    else:
                        o = env.reset()

            mean_episode_success_rate = success_rate / EPISODE_NUM_PER_CKPT
            mean_episode_reward = total_reward / EPISODE_NUM_PER_CKPT
            mean_episode_cost = total_cost / EPISODE_NUM_PER_CKPT
            mean_overtake_vehicle_num = total_overtake_vehicle_num / EPISODE_NUM_PER_CKPT
            mean_position = sum([ep["avg_position"] for ep in super_data[ckpt]]) / EPISODE_NUM_PER_CKPT

            # When writing data to CSV
            mean_velocity = sum([ep["avg_velocity"] for ep in super_data[ckpt]]) / EPISODE_NUM_PER_CKPT
            mean_acceleration = sum([ep["avg_acceleration"] for ep in super_data[ckpt]]) / EPISODE_NUM_PER_CKPT

            # Print to console
            print(
                "CKPT:{} | success_rate:{}, mean_episode_reward:{}, mean_episode_cost:{}, mean_velocity:{}, mean_acceleration:{}, mean_overtake_vehicle_num:{}, mean_position:{}".format(
                    ckpt,
                    mean_episode_success_rate,
                    mean_episode_reward,
                    mean_episode_cost,
                    mean_velocity,
                    mean_acceleration,
                    mean_overtake_vehicle_num,
                    mean_position
                ))

            # Write to CSV
            writer.writerow([ckpt, mean_episode_success_rate, mean_episode_reward, mean_episode_cost,
                             mean_velocity, mean_acceleration,
                             mean_overtake_vehicle_num, mean_position])

            del compute_actions

    env.close()

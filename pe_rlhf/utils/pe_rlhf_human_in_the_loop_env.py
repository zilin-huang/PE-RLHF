import copy

from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.engine.core.manual_controller import KeyboardController, SteeringWheelController
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.engine.engine_utils import get_global_config
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.utils.math_utils import safe_clip
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.policy.idm_policy import IDMPolicy
from pe_rlhf.utils.policy.ValueTakeoverPolicy import ValueControllableIDMPolicy,ValueControllableIDM_MOBILPolicy

from pe_rlhf.utils.policy.IDM_MOBIL_env import IDM_MOBILPolicy

import os.path as osp
import numpy as np
from os import scandir
from pe_rlhf.utils.saved_expert.save_expert import compress_model
from pe_rlhf.utils.common import load_weights

ScreenMessage.SCALE = 0.1

class HumanInTheLoopEnv(SafeMetaDriveEnv):
    """
    This Env depends on the new version of MetaDrive
    """

    steps = 0

    def default_config(self):
        config = super(HumanInTheLoopEnv, self).default_config()
        config.update(
            {
                "environment_num": 50,
                "start_seed": 100,
                "cost_to_reward": True,
                "traffic_density": 0.06,
                "manual_control": False,
                "controller": "joystick",
                "agent_policy": ValueControllableIDMPolicy,
                # "agent_policy": ValueControllableIDM_MOBILPolicy,
                "only_takeover_start_cost": True,
                "main_exp": True,
                "random_spawn": False,
                "cos_similarity": True,
                "out_of_route_done": True,
                "in_replay": False,
                "random_spawn_lane_index": False,
                "target_vehicle_configs": {
                    "default_agent": {"spawn_lane_index": (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 1)}},

                "dagger_takeover": False,  # new parameter
                "maxmin_takeover": False,  # new parameter
                "uncertainty_takeover": False,  # new parameter
                "ensemble": False,  # new parameter
                "value_takeover_threshold": 1.5,  # new parameter
                "var_threshold": 2.0,  # new parameter

                "warmup_ts": 100,  # new parameter

                "eval": False,  # new parameter

                "update_value_freq": 1000,  # new parameter
                "exp_path": None,  # new parameter
                "expert_value_weights": "default",  # new parameter
                "value_fn_path": None,   # new parameter
                "value_from_scratch": False,  # new parameter

                # If use the value_based_takeover_method
                "value_based_takeover": False,   # new parameter
            },
            allow_add_new_key=True
        )
        return config

    def __init__(self, config):
        super(HumanInTheLoopEnv, self).__init__(config)


    def _after_lazy_init(self):
        value_weight_path = osp.join(osp.dirname(__file__),
                                     "saved_expert",
                                     "sac_expert.npz") \
            if self.config["expert_value_weights"] == "default" \
            else self.config["expert_value_weights"]
        self.expert_value_weights = load_weights(value_weight_path) if value_weight_path is not None else None
        self.ensemble = self.config["ensemble"]

        self.latest_idx = 0  # for loading latest value weights
        self.max_idx_for_value = 50

        self.engine.value_based_takeover_method = self.value_based_takeover_method

    def reset(self, *args, **kwargs):
        self.in_stop = False
        self.t_o = False
        self.total_takeover_cost = 0
        self.input_action = None

        self.last_velocity = 0  # Initialize the speed of the previous frame to 0
        self.position = 0  # Initialize position at the start of each episode

        ret = super(HumanInTheLoopEnv, self).reset(*args, **kwargs)
        if self.config["random_spawn"]:
            self.config["vehicle_config"]["spawn_lane_index"] = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2,
                                                                 self.engine.np_random.randint(3))
        # keyboard is not as good as steering wheel, so set a small speed limit
        self.vehicle.update_config({"max_speed": 25 if self.config["controller"] == "keyboard" else 40})
        return ret

    def _get_step_return(self, actions, engine_info):
        o, r, d, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        if self.config["in_replay"]:
            return o, r, d, engine_info
        controller = self.engine.get_policy(self.vehicle.id)
        last_t = self.t_o
        self.t_o = controller.takeover if hasattr(controller, "takeover") else False
        engine_info["takeover_start"] = True if not last_t and self.t_o else False
        engine_info["takeover"] = self.t_o
        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.t_o

        if not condition:
            self.total_takeover_cost += 0
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost

        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info["cost"]
        engine_info["total_native_cost"] = self.episode_cost

        # Calculate the acceleration of the vehicle
        current_velocity = engine_info["velocity"] * (1 / 3.6)  # The initial speed is km/h, which is converted to m/s
        self.last_velocity = current_velocity
        time_step = 0.1
        self.position += self.last_velocity * time_step  # last_velocity is in m/s
        engine_info["current_position"] = self.position

        value_based_takeover, warmup = self.engine.get_policy(
            self.vehicle.id).value_based_takeover, self.engine.get_policy(self.vehicle.id).warmup
        engine_info["value_based_takeover"] = value_based_takeover
        engine_info["warmup"] = warmup

        # print("takeover_action:", engine_info["raw_action"])
        # print("input_action:", actions)
        # print("current_position", engine_info["current_position"])
        # print(engine_info)
        return o, r, d, engine_info

    def _is_out_of_road(self, vehicle):
        ret = (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    def step(self, actions):
        if not self.config["eval"]:
            if HumanInTheLoopEnv.steps % self.config["update_value_freq"] == 0 and self.config["value_from_scratch"]:
                self.load_latest_value_weights()

        self.input_action = copy.copy(actions)
        ret = super(HumanInTheLoopEnv, self).step(actions)

        if not self.config["eval"]:
            HumanInTheLoopEnv.steps += 1

        while self.in_stop:
            self.engine.taskMgr.step()
        if self.config["use_render"] and self.config["main_exp"] and not self.config["in_replay"]:
            super(HumanInTheLoopEnv, self).render(text={
                "Total Cost": self.episode_cost,
                "Takeover Cost": self.total_takeover_cost,
                "Takeover": self.t_o,
                "COST": ret[-1]["takeover_cost"],
                "Velocity": ret[-1]["velocity"],
                # "Stop (Press E)": "",
                "Warmup": ret[-1]["warmup"],
                "Value Takeover": ret[-1]["value_based_takeover"]
            })
        return ret

    def value_based_takeover_method(self, agent_id, idm_action, expert_action):
        threshold = self.engine.global_config["value_takeover_threshold"]
        warmup_ts = self.engine.global_config["warmup_ts"]
        value_based_takeover = False

        if HumanInTheLoopEnv.steps < warmup_ts and not self.engine.global_config["eval"]:
            value_based_takeover = True
            warmup = True
            final_action = expert_action
        else:
            warmup = False
            if not self.engine.global_config["ensemble"]:
                expert_value = self.get_q_value(agent_id, action=expert_action)
                idm_value = self.get_q_value(agent_id, action=idm_action, pessimistic=True)
                if idm_value < expert_value - threshold:
                    value_based_takeover = True
                # print("no_ensemble_takeover:", value_based_takeover)
            else:
                if self.engine.global_config["maxmin_takeover"]:
                    sampled_actions = [self.action_space.sample() for _ in range(10)]  # Sampling directly from action space
                    q_values = [self.get_q_value(agent_id, action=action, pessimistic=False) for action in
                                sampled_actions]
                    maxmin_diff = np.max(q_values) - np.min(q_values)
                    if maxmin_diff > threshold:
                        value_based_takeover = True
                    # print("maxmin_takeover:", value_based_takeover)
                elif self.engine.global_config["uncertainty_takeover"]:
                    expert_ensemble_values = self.get_q_value(agent_id, action=expert_action, ensemble=True)
                    expert_var = np.var(expert_ensemble_values)
                    if expert_var > self.engine.global_config["var_threshold"]:
                        value_based_takeover = True
                    # print("uncertainty_takeover:", value_based_takeover)
                else:
                    expert_ensemble_values = self.get_q_value(agent_id, action=expert_action, ensemble=True)
                    expert_var = np.var(expert_ensemble_values)
                    idm_ensemble_values = self.get_q_value(agent_id, action=idm_action, ensemble=True)
                    idm_var = np.var(idm_ensemble_values)
                    diff = np.array(expert_ensemble_values) - np.array(idm_ensemble_values)
                    diff_mean, diff_var = np.average(diff), np.var(diff)
                    if diff_mean > threshold or idm_var > 2 * self.engine.global_config["var_threshold"]:
                        value_based_takeover = True
                    # print("var_threshold:", value_based_takeover)

        if value_based_takeover:
            final_action = expert_action
        else:
            final_action = idm_action
        return value_based_takeover, final_action, warmup

    def get_q_value(self, agent_id, action, pessimistic=False, ensemble=False):
        obs = self.observations[agent_id].observe(self.vehicles[agent_id])
        weight = self.expert_value_weights
        action = np.array(action)  # Converting actions to NumPy arrays
        if ensemble:
            return self.ensemble_q_value(action, obs, weight, pessimistic=pessimistic)
        else:
            return self.expert_q_value(action, obs, weight, pessimistic=pessimistic)

    # Calculate the Q-value for a single action
    def expert_q_value(self, action, obs, weights, twin=False, pessimistic=False):
        if pessimistic:
            # A Boolean value indicating whether to compute a pessimistic (pessimistic) Q-value.
            # If set to True, it will return the smallest of the two Q-values
            return np.min((self.expert_q_value(action, obs, weights), self.expert_q_value(action, obs, weights, twin=True)))
        if twin:
            key_pre = "default_policy/sequential_2/twin_"
        else:
            key_pre = "default_policy/sequential_1/"
        obs = obs.reshape(1, -1)
        action = action.reshape(1, -1)
        input = np.hstack((obs, action))
        x = np.matmul(input, weights[key_pre + "q_hidden_0/kernel"]) + weights[key_pre + "q_hidden_0/bias"]
        x = self.relu(x)
        x = np.matmul(x, weights[key_pre + "q_hidden_1/kernel"]) + weights[key_pre + "q_hidden_1/bias"]
        x = self.relu(x)
        x = np.matmul(x, weights[key_pre + "q_out/kernel"]) + weights[key_pre + "q_out/bias"]

        return x

    def ensemble_q_value(self, action, obs, weights, twin=False, pessimistic=False):
        obs = obs.reshape(1, -1)
        action = np.array(action).reshape(1, -1)
        input = np.hstack((obs, action))
        values = []

        # for HACO expert
        if self.config["value_from_scratch"]:
            for key_pre in [
                "default_policy/sequential_1/q_",
                "default_policy/sequential_2/twin_q_",
                "default_policy/sequential_3/q_0_",
                "default_policy/sequential_4/q_1_",
                "default_policy/sequential_5/q_2_",
            ]:
                x = np.matmul(input, weights[key_pre + "hidden_0/kernel"]) + weights[key_pre + "hidden_0/bias"]
                x = self.relu(x)
                x = np.matmul(x, weights[key_pre + "hidden_1/kernel"]) + weights[key_pre + "hidden_1/bias"]
                x = self.relu(x)
                x = np.matmul(x, weights[key_pre + "out/kernel"]) + weights[key_pre + "out/bias"]
                values.append(x[0][0])

        # for SAC expert
        else:
            for key_pre in [
                "default_policy/sequential_1/q_",
                "default_policy/sequential_2/twin_q_",
                "default_policy/sequential_4/q_",
                "default_policy/sequential_5/twin_q_",
            ]:
                x = np.matmul(input, weights[key_pre + "hidden_0/kernel"]) + weights[key_pre + "hidden_0/bias"]
                x = self.relu(x)
                x = np.matmul(x, weights[key_pre + "hidden_1/kernel"]) + weights[key_pre + "hidden_1/bias"]
                x = self.relu(x)
                x = np.matmul(x, weights[key_pre + "out/kernel"]) + weights[key_pre + "out/bias"]
                values.append(x[0][0])

        return values

    def relu(self, x):
        return (np.abs(x) + x) / 2

    def load_latest_value_weights(self):
        if self.config["value_fn_path"] is not None:
            self.expert_value_weights = compress_model(self.config["value_fn_path"])
        else:
            exp_dir = self.config["exp_path"]
            trainer_subfolders = [f.path for f in scandir(exp_dir) if f.is_dir()]

            if len(trainer_subfolders) == 0:
                return

            # Flatten the list of subfolders to get all checkpoints in all trainer subfolders
            ckpt_paths = []
            for trainer_subfolder in trainer_subfolders:
                ckpt_subfolders = [f.path for f in scandir(trainer_subfolder) if f.is_dir()]
                for ckpt_subfolder in ckpt_subfolders:
                    ckpt_idx = int(ckpt_subfolder.split("_")[-1])
                    ckpt_paths.append((ckpt_idx, ckpt_subfolder))

            if len(ckpt_paths) == 0:
                return

            # Find the latest checkpoint
            latest_idx, latest_ckpt_path = max(ckpt_paths, key=lambda x: x[0])

            if self.max_idx_for_value > latest_idx >= self.latest_idx:
                self.latest_idx = latest_idx
                self.expert_value_weights = compress_model(
                    osp.join(latest_ckpt_path, "checkpoint-%d" % latest_idx))

    def stop(self):
        self.in_stop = not self.in_stop

    def setup_engine(self):
        super(HumanInTheLoopEnv, self).setup_engine()
        self.engine.accept("e", self.stop)

    def get_takeover_cost(self, info):
        if not self.config["cos_similarity"]:
            return 1
        takeover_action = np.clip(np.array(info["raw_action"]), -1, 1)
        agent_action = np.clip(np.array(self.input_action), -1, 1)

        multiplier = np.dot(agent_action, takeover_action)
        divident = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        cos_dist = multiplier / divident if divident >= 1e-6 else 1.0

        return 1 - cos_dist


if __name__ == "__main__":
    env = HumanInTheLoopEnv(
        {"manual_control": True,
         "disable_model_compression": True,
         "use_render": True,
         "main_exp": True,
         "controller": "keyboard",
         "ensemble": True,
         # "maxmin_takeover": True,  # Ensure maxmin_takeover is set to True
         "agent_policy": ValueControllableIDM_MOBILPolicy}
    )
    env.reset()
    while True:
        env.step([0, 0])


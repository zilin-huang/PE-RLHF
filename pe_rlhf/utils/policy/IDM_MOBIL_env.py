from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive import SafeMetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy

import sys
import numpy as np
from metadrive.policy.idm_policy import FrontBackObjects

class IDM_MOBILPolicy(IDMPolicy):
    """
    The IDM_MOBILPolicy class inherits from the IDMPolicy class and overrides the lane change policy, fully based on the MOBIL model.
    """

    # Define MOBIL model parameters
    POLITENESS = 0.1  # Politeness factor
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # Minimum acceleration gain for lane change
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # Maximum braking imposed by lane change
    LANE_CHANGE_DELAY = 1.0  # Lane change delay time

    # Define IDM model parameters
    COMFORT_ACC_MAX = 2.0  # Maximum comfortable acceleration
    COMFORT_ACC_MIN = -5.0  # Maximum comfortable deceleration
    DISTANCE_WANTED = 10.0  # Desired distance
    TIME_WANTED = 1.5  # Desired headway time
    DELTA = 4.0  # Acceleration exponent

    def __init__(self, control_object, random_seed):
        super(IDM_MOBILPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.action_info = {}  # Initialize action_info dictionary

    def act(self, *args, **kwargs):
        # Concatenate lane
        success = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        try:
            if success and self.enable_lane_change:
                # Perform lane change due to routing
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                # Cannot find routing target lane
                surrounding_objects = FrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object.position,
                    max_distance=self.MAX_LONG_DIST
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
                self.action_info["lane_change"] = False  # Default to no lane change
        except Exception as e:
            # Print detailed error information
            # print(f"IDM_MOBILPolicy error: {e}")
            # print(f"Error occurred at line: {sys.exc_info()[-1].tb_lineno}")
            # print(f"Variable values: success={success}, self.enable_lane_change={self.enable_lane_change}")

            # Error fallback
            acc_front_obj = None
            acc_front_dist = 5
            steering_target_lane = self.routing_target_lane
            self.action_info["lane_change"] = False

        # Control by PID and IDM
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)
        action = [steering, acc]

        # Add action information
        self.action_info["steering"] = steering
        self.action_info["acceleration"] = acc

        return action

    def lane_change_policy(self, all_objects):
        # print(f"Entering lane_change_policy, all_objects: {all_objects}")
        current_lanes = self.control_object.navigation.current_ref_lanes

        # Determine whether to change to the left lane based on the MOBIL model
        left_lane_index = current_lanes.index(self.routing_target_lane) - 1
        if left_lane_index >= 0:
            # print(f"Checking left lane {current_lanes[left_lane_index]}")
            if self.mobil(current_lanes[left_lane_index]):
                # print("Changing to left lane")
                return self.change_lane(all_objects, current_lanes[left_lane_index], True)  # Change to the left lane

        # Determine whether to change to the right lane based on the MOBIL model
        right_lane_index = current_lanes.index(self.routing_target_lane) + 1
        if right_lane_index < len(current_lanes):
            # print(f"Checking right lane {current_lanes[right_lane_index]}")
            if self.mobil(current_lanes[right_lane_index]):
                # print("Changing to right lane")
                return self.change_lane(all_objects, current_lanes[right_lane_index], True)  # Change to the right lane

        # If the MOBIL model conditions are not met, keep the current lane
        # print("Keeping current lane")
        self.target_speed = self.NORMAL_SPEED
        surrounding_objects = FrontBackObjects.get_find_front_back_objs(
            all_objects,
            self.routing_target_lane,
            self.control_object.position,
            max_distance=self.MAX_LONG_DIST
        )
        self.action_info["lane_change"] = False  # Set lane_change to False
        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane

    def change_lane(self, all_objects, target_lane, lane_change=False):
        # Perform lane change
        self.target_speed = self.NORMAL_SPEED
        surrounding_objects = FrontBackObjects.get_find_front_back_objs(
            all_objects,
            target_lane,
            self.control_object.position,
            max_distance=self.MAX_LONG_DIST
        )
        self.overtake_timer = 0
        self.action_info["lane_change"] = lane_change  # Set lane_change to the passed value
        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), target_lane

    def mobil(self, lane):
        # print(f"Entering MOBIL for lane {lane}")

        try:
            # Calculate acceleration in the MOBIL model
            new_obs = FrontBackObjects.get_find_front_back_objs(
                self.control_object.lidar.get_surrounding_objects(self.control_object),
                lane,
                self.control_object.position,
                max_distance=self.MAX_LONG_DIST
            )

            if self.control_object is None:
                # print("Warning: self.control_object is None in mobil()")
                return False

            new_following = new_obs.back_object()
            if new_following is None:
                # print("Warning: new_following is None in mobil(), assuming no impact on lane change")
                new_following_a = 0
                new_following_pred_a = 0
            else:
                new_following_a = self.acceleration(new_following, self.desired_gap(new_following, self.control_object))

                new_preceding = new_obs.front_object()
                if new_preceding is None:
                    # print("Warning: new_preceding is None in mobil()")
                    return False

                new_following_pred_a = self.acceleration(new_following, self.desired_gap(new_following, new_preceding))

            old_obs = FrontBackObjects.get_find_front_back_objs(
                self.control_object.lidar.get_surrounding_objects(self.control_object),
                self.routing_target_lane,
                self.control_object.position,
                max_distance=self.MAX_LONG_DIST
            )

            if self.control_object is None:
                # print("Warning: self.control_object is None in mobil()")
                return False

            old_preceding = old_obs.front_object()
            if old_preceding is None:
                # print("Warning: old_preceding is None in mobil(), assuming no impact on lane change")
                self_a = self.ACC_FACTOR  # Assume no obstacle in the current lane, maintaining high acceleration
            else:
                self_a = self.acceleration(self.control_object, self.desired_gap(self.control_object, old_preceding))

            old_following = old_obs.back_object()
            if old_following is None:
                # print("Warning: old_following is None in mobil(), assuming no impact on lane change")
                old_following_a = 0
                old_following_pred_a = 0
            else:
                old_following_a = self.acceleration(old_following, self.desired_gap(old_following, self.control_object))
                old_following_pred_a = self.acceleration(old_following, self.desired_gap(old_following, old_preceding))

            new_preceding = new_obs.front_object()
            if new_preceding is None:
                # print("Warning: new_preceding is None in mobil(), assuming no impact on lane change")
                self_pred_a = 0
            else:
                self_pred_a = self.acceleration(self.control_object, self.desired_gap(self.control_object, new_preceding))

            # Calculate lane change benefit using the MOBIL model formula
            jerk = self_pred_a - self_a + self.POLITENESS * (
                    new_following_pred_a - new_following_a + old_following_pred_a - old_following_a
            )

            if jerk > self.LANE_CHANGE_MIN_ACC_GAIN:
                # print(f"MOBIL condition satisfied with jerk {jerk}")
                return True
            else:
                # print(f"MOBIL condition not satisfied with jerk {jerk}")
                return False

        except Exception as e:
            # print(f"Error in MOBIL: {e}")
            return False

    def desired_gap(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_obj.velocity, ego_vehicle.heading) if projected \
            else ego_vehicle.speed - front_obj.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star




if __name__ == "__main__":
    env = SafeMetaDriveEnv(
        {
            "traffic_density": 0.06,
            "manual_control": True,
            "use_render": True,
            'environment_num': 10,
            "start_seed": 129,
            "controller": "keyboard",
            "agent_policy": IDM_MOBILPolicy,
            "vehicle_config": {
                "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 2),
            }
        }
    )

    o = env.reset()

    # Create an instance of IDM_MOBILPolicy after reset
    agent = IDM_MOBILPolicy(env.vehicle, env.current_seed)

    total_cost = 0
    # In the main loop
    for i in range(1, 100000):
        # Pass agent.act() to env.step()
        o, r, d, info = env.step(agent.act())
        total_cost += info["cost"]
        env.render(
            text={
                "cost": total_cost,
                "seed": env.current_seed,
                # "reward": r,
                "total_cost": info["total_cost"],
                "lane_change": agent.action_info["lane_change"],
                "steering": agent.action_info["steering"],
                "acceleration": agent.action_info["acceleration"]
            }
        )
        if info["crash_vehicle"]:
            print("crash_vehicle: cost {}, reward {}".format(info["cost"], r))
        if info["crash_object"]:
            print("crash_object: cost {}, reward {}".format(info["cost"], r))

        # Uncomment the following lines to reset the environment when done
        # if d:
        #     total_cost = 0
        #     print("done_cost: {}".format(info["cost"]), "done_reward: {}".format(r))
        #     print("Reset")
        #     env.reset()
    env.close()

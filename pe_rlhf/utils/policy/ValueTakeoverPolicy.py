from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.engine.core.manual_controller import KeyboardController, SteeringWheelController
from metadrive.engine.engine_utils import get_global_config
from metadrive.policy.idm_policy import IDMPolicy
from pe_rlhf.utils.policy.IDM_MOBIL_env import IDM_MOBILPolicy


class TakeoverPolicy(EnvInputPolicy):
    """
    Record the takeover signal and incorporate IDMPolicy actions.
    """

    def __init__(self, obj, seed):
        super(TakeoverPolicy, self).__init__(obj, seed)
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            if config["controller"] == "joystick":
                self.controller = SteeringWheelController()
            elif config["controller"] == "keyboard":
                self.controller = KeyboardController(False)
            else:
                raise ValueError("Unknown Policy: {}".format(config["controller"]))
        self.takeover = False

        self.idm_policy = IDMPolicy(control_object=obj, random_seed=seed)  # Initialize IDMPolicy


    def act(self, agent_id):
        # Get action from IDMPolicy
        idm_action = self.idm_policy.act(agent_id)
        # print(idm_action)

        agent_action = super(TakeoverPolicy, self).act(agent_id)
        # print(agent_action)

        # Check for expert takeover
        if self.engine.global_config["manual_control"] and self.engine.agent_manager.get_agent(
                agent_id) is self.engine.current_track_vehicle and not self.engine.main_camera.is_bird_view_camera():
            expert_action = self.controller.process_input(self.engine.current_track_vehicle)
            if isinstance(self.controller, SteeringWheelController) and (self.controller.left_shift_paddle
                                                                         or self.controller.right_shift_paddle):
                self.takeover = True
                return expert_action
            elif isinstance(self.controller, KeyboardController) and abs(sum(expert_action)) > 1e-2:
                self.takeover = True
                return expert_action

        # If no takeover, return IDM action
        self.takeover = False
        return agent_action


class MyKeyboardController(KeyboardController):
    # Update Parameters
    STEERING_INCREMENT = 0.05
    STEERING_DECAY = 0.5

    THROTTLE_INCREMENT = 0.5
    THROTTLE_DECAY = 1

    BRAKE_INCREMENT = 0.5
    BRAKE_DECAY = 1

class ValueControllableIDMPolicy(EnvInputPolicy):
    """
    Determine whether the human expert's action or the IDM_MOBILPolicy's action is used when the human expert takes over
    use the Agent's action when it doesn't.
    """
    def __init__(self, obj, seed):
        super(ValueControllableIDMPolicy, self).__init__(obj, seed)
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            if config["controller"] == "joystick":
                self.controller = SteeringWheelController()
            elif config["controller"] == "keyboard":
                self.controller = MyKeyboardController(False)
            else:
                raise ValueError("Unknown Policy: {}".format(config["controller"]))
        self.takeover = False
        self.takeover = False
        self.value_based_takeover = False
        self.warmup = False
        self.idm_policy = IDMPolicy(control_object=obj, random_seed=seed)  # Initialize IDMPolicy

    def act(self, agent_id):
        # Get action from IDMPolicy
        idm_action = self.idm_policy.act(agent_id)
        # print(idm_action)

        agent_action = super(ValueControllableIDMPolicy, self).act(agent_id)
        # print(agent_action)

        # Check for expert takeover
        if self.engine.global_config["manual_control"] and self.engine.agent_manager.get_agent(
                agent_id) is self.engine.current_track_vehicle and not self.engine.main_camera.is_bird_view_camera():
            expert_action = self.controller.process_input(self.engine.current_track_vehicle)
            if isinstance(self.controller, SteeringWheelController) and (self.controller.left_shift_paddle
                                                                         or self.controller.right_shift_paddle):
                self.takeover = True
                self.value_based_takeover, final_action, self.warmup = self.engine.value_based_takeover_method(agent_id,
                                                                                                               idm_action,
                                                                                                               expert_action)
                return final_action
            elif isinstance(self.controller, MyKeyboardController) and abs(sum(expert_action)) > 1e-2:
                self.takeover = True
                self.value_based_takeover, final_action, self.warmup = self.engine.value_based_takeover_method(agent_id, idm_action,
                                                                                            expert_action)
                return final_action

        # If no takeover, return IDM action
        self.takeover = False
        self.value_based_takeover = False
        return agent_action


class ValueControllableIDM_MOBILPolicy(EnvInputPolicy):
    """
    Determine whether the human expert's action or the IDM_MOBILPolicy's action is used when the human expert takes over
    use the Agent's action when it doesn't.
    """
    def __init__(self, obj, seed):
        super(ValueControllableIDM_MOBILPolicy, self).__init__(obj, seed)
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            if config["controller"] == "joystick":
                self.controller = SteeringWheelController()
            elif config["controller"] == "keyboard":
                self.controller = MyKeyboardController(False)
            else:
                raise ValueError("Unknown Policy: {}".format(config["controller"]))
        self.takeover = False
        self.takeover = False
        self.value_based_takeover = False
        self.warmup = False
        self.idm_mobil_policy = IDM_MOBILPolicy(control_object=obj, random_seed=seed)  # Initialize IDMPolicy

    def act(self, agent_id):
        # Get action from IDMPolicy
        idm_mobil_action = self.idm_mobil_policy.act(agent_id)
        # print(idm_action)

        agent_action = super(ValueControllableIDM_MOBILPolicy, self).act(agent_id)
        # print(agent_action)

        # Check for expert takeover
        if self.engine.global_config["manual_control"] and self.engine.agent_manager.get_agent(
                agent_id) is self.engine.current_track_vehicle and not self.engine.main_camera.is_bird_view_camera():
            expert_action = self.controller.process_input(self.engine.current_track_vehicle)
            if isinstance(self.controller, SteeringWheelController) and (self.controller.left_shift_paddle
                                                                         or self.controller.right_shift_paddle):
                self.takeover = True
                self.value_based_takeover, final_action, self.warmup = self.engine.value_based_takeover_method(agent_id,
                                                                                                               idm_mobil_action,
                                                                                                               expert_action)
                return final_action
            elif isinstance(self.controller, MyKeyboardController) and abs(sum(expert_action)) > 1e-2:
                self.takeover = True
                self.value_based_takeover, final_action, self.warmup = self.engine.value_based_takeover_method(agent_id, idm_mobil_action,
                                                                                            expert_action)
                return final_action

        # If no takeover, return IDM action
        self.takeover = False
        self.value_based_takeover = False
        return agent_action

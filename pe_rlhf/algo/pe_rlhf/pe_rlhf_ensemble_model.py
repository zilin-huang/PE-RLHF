from gym.spaces import Discrete
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()
import numpy as np
from pe_rlhf.algo.ensembleQ.ensembleQ_model import EnsembleQ_model
from ray.rllib.models import ModelCatalog
import logging
from ray.rllib.agents.sac.sac_torch_model import SACTorchModel

logger = logging.getLogger(__name__)


class PE_RLHFEnsembleModel(EnsembleQ_model):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 actor_hidden_activation="relu",
                 actor_hiddens=(256, 256),
                 critic_hidden_activation="relu",
                 critic_hiddens=(256, 256),
                 twin_cost_q=False,  # NEW
                 twin_q=False,
                 initial_alpha=1.0,
                 target_entropy=None):
        super(PE_RLHFEnsembleModel, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
            actor_hidden_activation=actor_hidden_activation,
            actor_hiddens=actor_hiddens,
            critic_hidden_activation=critic_hidden_activation,
            critic_hiddens=critic_hiddens,
            twin_q=twin_q,
            initial_alpha=initial_alpha,
            target_entropy=target_entropy,
        )
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.discrete = True
            action_outs = q_outs = self.action_dim
        else:
            self.action_dim = np.product(action_space.shape)
            self.discrete = False
            action_outs = 2 * self.action_dim
            q_outs = 1

        def build_q_net(name, observations, actions):
            # For continuous actions: Feed obs and actions (concatenated)
            # through the NN. For discrete actions, only obs.
            q_net = tf.keras.Sequential(([
                                             tf.keras.layers.Concatenate(axis=1),
                                         ] if not self.discrete else []) + [
                                            tf.keras.layers.Dense(
                                                units=units,
                                                activation=getattr(tf.nn, critic_hidden_activation, None),
                                                name="{}_hidden_{}".format(name, i))
                                            for i, units in enumerate(critic_hiddens)
                                        ] + [
                                            tf.keras.layers.Dense(
                                                units=q_outs, activation=None, name="{}_out".format(name))
                                        ])

            # TODO(hartikainen): Remove the unnecessary Model calls here
            if self.discrete:
                q_net = tf.keras.Model(observations, q_net(observations))
            else:
                q_net = tf.keras.Model([observations, actions],
                                       q_net([observations, actions]))
            return q_net

        # Added the following code
        self.cost_q_net = build_q_net(
            "cost_q", self.model_out, self.actions_input
        )
        self.register_variables(self.cost_q_net.variables)

        if twin_cost_q:
            self.cost_twin_q_net = build_q_net(
                "cost_twin_q", self.model_out, self.actions_input
            )
            self.register_variables(self.cost_twin_q_net.variables)
        else:
            self.cost_twin_q_net = None

    # Added the following code
    def get_cost_q_values(self, model_out, actions=None):
        if actions is not None:
            return self.cost_q_net([model_out, actions])
        else:
            return self.cost_q_net(model_out)

    # Added the following code
    def get_twin_cost_q_values(self, model_out, actions=None):
        if actions is not None:
            return self.cost_twin_q_net([model_out, actions])
        else:
            return self.cost_twin_q_net(model_out)

    # Added the following code
    def cost_q_variables(self):
        return self.cost_q_net.variables + (
            self.cost_twin_q_net.variables
            if self.cost_twin_q_net else []
        )


# The ensembleQ_policy also has this function, and it is basically the same
def build_ensembleQ_model(policy, obs_space, action_space, config):
    # 2 cases:
    # 1) with separate state-preprocessor (before obs+action concat).
    # 2) no separate state-preprocessor: concat obs+actions right away.
    if config["use_state_preprocessor"]:
        num_outputs = 256  # Flatten last Conv2D to this many nodes.
    else:
        num_outputs = 0
        # No state preprocessor: fcnet_hiddens should be empty.
        if config["model"]["fcnet_hiddens"]:
            logger.warning(
                "When not using a state-preprocessor with SAC, `fcnet_hiddens`"
                " will be set to an empty list! Any hidden layer sizes are "
                "defined via `policy_model.fcnet_hiddens` and "
                "`Q_model.fcnet_hiddens`.")
            config["model"]["fcnet_hiddens"] = []

    # Force-ignore any additionally provided hidden layer sizes.
    # Everything should be configured using SAC's "Q_model" and "policy_model"
    # settings.
    policy.model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=SACTorchModel if config["framework"] == "torch" else PE_RLHFEnsembleModel,
        name="sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        twin_cost_q=config["twin_cost_q"],  # NEW code
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"])

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=SACTorchModel if config["framework"] == "torch" else PE_RLHFEnsembleModel,
        name="target_sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        twin_cost_q=config["twin_cost_q"],  # NEW code
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"])

    return policy.model

from gym.spaces import Discrete
import numpy as np

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()
from ray.rllib.agents.sac.sac_tf_model import SACTFModel

# Number of Q networks in addition to the original Q and twin Q networks
# 5 Q networks in total
ENSEMBLE_CNT = 3  # In addition to the standard Q network (original Q) and twin Q network, there are 3 extra Q networks
class EnsembleQ_model(SACTFModel):

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
                 twin_q=False,
                 initial_alpha=1.0,
                 target_entropy=None):

        super(EnsembleQ_model, self).__init__(
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

        # Modified the original code: computing Q values
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

        # Introduce ensembled Q net
        self.ensemble_q_net = []
        for i in range(ENSEMBLE_CNT):
            q_net = build_q_net("q_{}".format(i), self.model_out, self.actions_input)
            self.register_variables(q_net.variables)
            self.ensemble_q_net.append(q_net)  # A list named ensemble_q_net is used, containing multiple Q networks

    def q_variables(self):
        q_vars = super(EnsembleQ_model, self).q_variables()
        for q_net in self.ensemble_q_net:
            q_vars += q_net.variables
        return q_vars

    # This code returns more Q value estimates, including the original Q value, twin Q value, and extra Q values
    def get_all_q_values(self, model_out, actions=None):
        q_value = super(EnsembleQ_model, self).get_q_values(model_out, actions)
        twin_q_value = super(EnsembleQ_model, self).get_twin_q_values(model_out, actions)
        extra_q_values = self.get_extra_q_values(model_out, actions)

        return [q_value, twin_q_value, *extra_q_values]

    # This method is used to get extra Q value estimates, which come from multiple Q networks in the ensemble Q network
    def get_extra_q_values(self, model_out, actions=None):
        q_values = []
        for q_net in self.ensemble_q_net:
            if actions is not None:
                q_values.append(q_net([model_out, actions]))
            else:
                q_values.append(q_net(model_out))

        return q_values

from gym.spaces import Box, Discrete
import logging

import ray
import ray.experimental.tf_utils
from ray.rllib.agents.ddpg.ddpg_tf_policy import ComputeTDErrorMixin, \
    TargetNetworkMixin
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio
from ray.rllib.agents.sac.sac_tf_model import SACTFModel
from ray.rllib.agents.sac.sac_torch_model import SACTorchModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import Beta, Categorical, \
    DiagGaussian, SquashedGaussian
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf, \
    try_import_tfp
from pe_rlhf.algo.ensembleQ.ensembleQ_model import ENSEMBLE_CNT
tf1, tf, tfv = try_import_tf()
tfp = try_import_tfp()

logger = logging.getLogger(__name__)

from ray.rllib.agents.sac.sac_tf_policy import SACTFPolicy, get_dist_class, ActorCriticOptimizerMixin
from pe_rlhf.algo.ensembleQ.ensembleQ_model import EnsembleQ_model

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
        model_interface=SACTorchModel  # Another one is: model_interface=ConstrainedSACModel
		#* change to use ensembleQ model
		# If the value of config["framework"] is "torch", choose to use SACTorchModel
		# If the value of config["framework"] is not "torch", choose to use EnsembleQ_model
		if config["framework"] == "torch" else EnsembleQ_model,
        name="sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"])

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=SACTorchModel   # Another one is: model_interface=ConstrainedSACModel
        if config["framework"] == "torch" else EnsembleQ_model,
        name="target_sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"])

    return policy.model

def sac_actor_ensemble_critic_loss(policy, model, _, train_batch):
    # Should be True only for debugging purposes (e.g. test cases)!
	deterministic = policy.config["_deterministic_loss"]

	model_out_t, _ = model({
        "obs": train_batch[SampleBatch.CUR_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

	model_out_tp1, _ = model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

	target_model_out_tp1, _ = policy.target_model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    # Discrete case.
	if model.discrete:
        # Get all action probs directly from pi and form their logp.
		# not ready
		assert False
		log_pis_t = tf.nn.log_softmax(model.get_policy_output(model_out_t), -1)
		policy_t = tf.math.exp(log_pis_t)
		log_pis_tp1 = tf.nn.log_softmax(
            model.get_policy_output(model_out_tp1), -1)
		policy_tp1 = tf.math.exp(log_pis_tp1)
        # Q-values.
		q_t = model.get_q_values(model_out_t)
        # Target Q-values.
		q_tp1 = policy.target_model.get_q_values(target_model_out_tp1)
		if policy.config["twin_q"]:
			twin_q_t = model.get_twin_q_values(model_out_t)
			twin_q_tp1 = policy.target_model.get_twin_q_values(
                target_model_out_tp1)
			q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)
		q_tp1 -= model.alpha * log_pis_tp1

		# Actually selected Q-values (from the actions batch).
		one_hot = tf.one_hot(
			train_batch[SampleBatch.ACTIONS], depth=q_t.shape.as_list()[-1])
		q_t_selected = tf.reduce_sum(q_t * one_hot, axis=-1)
		if policy.config["twin_q"]:
			twin_q_t_selected = tf.reduce_sum(twin_q_t * one_hot, axis=-1)
		# Discrete case: "Best" means weighted by the policy (prob) outputs.
		q_tp1_best = tf.reduce_sum(tf.multiply(policy_tp1, q_tp1), axis=-1)
		q_tp1_best_masked = \
            (1.0 - tf.cast(train_batch[SampleBatch.DONES], tf.float32)) * \
            q_tp1_best
    # Continuous actions case.
	else:
		# Sample simgle actions from distribution.
		action_dist_class = get_dist_class(policy.config, policy.action_space)
		action_dist_t = action_dist_class(
			model.get_policy_output(model_out_t), policy.model)
		policy_t = action_dist_t.sample() if not deterministic else \
			action_dist_t.deterministic_sample()
		log_pis_t = tf.expand_dims(action_dist_t.logp(policy_t), -1)
		action_dist_tp1 = action_dist_class(
			model.get_policy_output(model_out_tp1), policy.model)
		policy_tp1 = action_dist_tp1.sample() if not deterministic else \
			action_dist_tp1.deterministic_sample()
		log_pis_tp1 = tf.expand_dims(action_dist_tp1.logp(policy_tp1), -1)


		###################################
		# Equivalent to not calculating cost q below
		# Q-values for the actually selected actions.
		q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
		if policy.config["twin_q"]:
			twin_q_t = model.get_twin_q_values(
				model_out_t, train_batch[SampleBatch.ACTIONS])
			# Compared to Haco, this is the newly added code
			extra_q_ts = model.get_extra_q_values(
				model_out_t, train_batch[SampleBatch.ACTIONS])

        # Q-values for current policy in given current state.
		q_t_det_policy = model.get_q_values(model_out_t, policy_t)
		if policy.config["twin_q"]:
			twin_q_t_det_policy = model.get_twin_q_values(
				model_out_t, policy_t)
			# NEW code
			extra_q_t_det_policy = model.get_extra_q_values(
				model_out_t, policy_t)
			# Compare all Q value estimates (including current Q, double Q, and additional Q)
			# and choose the smallest Q value
			q_t_det_policy = tf.reduce_min(
				(q_t_det_policy, twin_q_t_det_policy, *extra_q_t_det_policy), axis=0)

        # target q network evaluation
		q_tp1 = policy.target_model.get_q_values(target_model_out_tp1,
													policy_tp1)
		if policy.config["twin_q"]:
			twin_q_tp1 = policy.target_model.get_twin_q_values(
				target_model_out_tp1, policy_tp1)
			# NEW code
			extra_q_tp1 = policy.target_model.get_extra_q_values(
				target_model_out_tp1, policy_tp1)
			# Take min over both twin-NNs.
			# TODO: randomly select two qs
			q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)

		q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
		if policy.config["twin_q"]:
			twin_q_t_selected = tf.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
			# NEW code
			extra_qs_selected = [tf.squeeze(i, axis=len(q_t.shape) - 1) for i in extra_q_ts]
		q_tp1 -= model.alpha * log_pis_tp1

		q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
		q_tp1_best_masked = (1.0 - tf.cast(train_batch[SampleBatch.DONES],
                                           tf.float32)) * q_tp1_best

	# compute RHS of bellman equation
	q_t_selected_target = tf.stop_gradient(
		train_batch[SampleBatch.REWARDS] +
		policy.config["gamma"]**policy.config["n_step"] * q_tp1_best_masked)

	# Compute the TD-error (potentially clipped).
	base_td_error = tf.math.abs(q_t_selected - q_t_selected_target)
	if policy.config["twin_q"]:
		twin_td_error = tf.math.abs(twin_q_t_selected - q_t_selected_target)
		# NEW code
		extra_td_error = 0
		for extra_q_selected in extra_qs_selected:
			extra_td_error += tf.math.abs(extra_q_selected - q_t_selected_target)
		td_error = 0.5 * (base_td_error + twin_td_error + extra_td_error)
	else:
		td_error = base_td_error

	critic_loss = [
		0.5 * tf.keras.losses.MSE(
			y_true=q_t_selected_target, y_pred=q_t_selected)
	]
	if policy.config["twin_q"]:
		critic_loss.append(0.5 * tf.keras.losses.MSE(
			y_true=q_t_selected_target, y_pred=twin_q_t_selected))
		# NEW code
		for extra_q_t_selected in extra_qs_selected:
			critic_loss.append(0.5 * tf.keras.losses.MSE(
				y_true=q_t_selected_target, y_pred=extra_q_t_selected
			))

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
	if model.discrete:
		assert False
		alpha_loss = tf.reduce_mean(
			tf.reduce_sum(
				tf.multiply(
					tf.stop_gradient(policy_t), -model.log_alpha *
					tf.stop_gradient(log_pis_t + model.target_entropy)),
				axis=-1))
		actor_loss = tf.reduce_mean(
			tf.reduce_sum(
				tf.multiply(
					# NOTE: No stop_grad around policy output here
					# (compare with q_t_det_policy for continuous case).
					policy_t,
					model.alpha * log_pis_t - tf.stop_gradient(q_t)),
				axis=-1))
	else:
		# No reward_loss
		# No cost_loss
		alpha_loss = -tf.reduce_mean(
			model.log_alpha *
			tf.stop_gradient(log_pis_t + model.target_entropy))
		actor_loss = tf.reduce_mean(model.alpha * log_pis_t - q_t_det_policy)

	# Save for stats function
	# No cost_loss, no reward_loss, no mean_batch_cost, and no cost-related parts
	policy.policy_t = policy_t
	policy.q_t = q_t
	policy.td_error = td_error
	policy.actor_loss = actor_loss
	policy.critic_loss = critic_loss
	policy.alpha_loss = alpha_loss
	policy.alpha_value = model.alpha
	policy.target_entropy = model.target_entropy

    # in a custom apply op we handle the losses separately, but return them
    # combined in one loss for now
	return actor_loss + tf.math.add_n(critic_loss) + alpha_loss

 
def gradients_fn(policy, optimizer, loss):
    # Eager: Use GradientTape.
	if policy.config["framework"] in ["tf2", "tfe"]:
		assert False
		tape = optimizer.tape
		pol_weights = policy.model.policy_variables()
		actor_grads_and_vars = list(
            zip(tape.gradient(policy.actor_loss, pol_weights), pol_weights))
		q_weights = policy.model.q_variables()
		if policy.config["twin_q"]:
			half_cutoff = len(q_weights) // 2
			grads_1 = tape.gradient(policy.critic_loss[0],
                                    q_weights[:half_cutoff])
			grads_2 = tape.gradient(policy.critic_loss[1],
                                    q_weights[half_cutoff:])
			critic_grads_and_vars = \
                list(zip(grads_1, q_weights[:half_cutoff])) + \
                list(zip(grads_2, q_weights[half_cutoff:]))
		else:
			critic_grads_and_vars = list(
				zip(
                    tape.gradient(policy.critic_loss[0], q_weights),
                    q_weights))

		alpha_vars = [policy.model.log_alpha]
		alpha_grads_and_vars = list(
            zip(tape.gradient(policy.alpha_loss, alpha_vars), alpha_vars))
    # Tf1.x: Use optimizer.compute_gradients()
	else:
		actor_grads_and_vars = policy._actor_optimizer.compute_gradients(
            policy.actor_loss, var_list=policy.model.policy_variables())

		# No cost-related parts
		q_weights = policy.model.q_variables()
		if policy.config["twin_q"]:
			cutoff = len(q_weights) // (2 + ENSEMBLE_CNT) # Originally, it was half_cutoff = len(q_weights) // 2
			# Modified the code below
			base_q_optimizer, twin_q_optimizer, *extra_q_optimizers = policy._critic_optimizer
			critic_grads_and_vars = base_q_optimizer.compute_gradients(
                policy.critic_loss[0], var_list=q_weights[:cutoff]
            ) + twin_q_optimizer.compute_gradients(
                policy.critic_loss[1], var_list=q_weights[cutoff:2 * cutoff])
			for i in range(ENSEMBLE_CNT):
				critic_grads_and_vars += extra_q_optimizers[i].compute_gradients(
					policy.critic_loss[2+i], var_list=q_weights[(2+i) * cutoff:(3+i) * cutoff]
				)
		else:
			critic_grads_and_vars = policy._critic_optimizer[
                0].compute_gradients(
                    policy.critic_loss[0], var_list=q_weights)
		alpha_grads_and_vars = policy._alpha_optimizer.compute_gradients(
            policy.alpha_loss, var_list=[policy.model.log_alpha])

    # Clip if necessary.
	if policy.config["grad_clip"]:
		clip_func = tf.clip_by_norm
	else:
		clip_func = tf.identity

    # Save grads and vars for later use in `build_apply_op`.
	policy._actor_grads_and_vars = [(clip_func(g), v)
									for (g, v) in actor_grads_and_vars
									if g is not None]
	policy._critic_grads_and_vars = [(clip_func(g), v)
										for (g, v) in critic_grads_and_vars
										if g is not None]
	policy._alpha_grads_and_vars = [(clip_func(g), v)
                                    for (g, v) in alpha_grads_and_vars
                                    if g is not None]
	# No cost parts
	grads_and_vars = (
        policy._actor_grads_and_vars + policy._critic_grads_and_vars +
        policy._alpha_grads_and_vars)
	return grads_and_vars

# Still no cost parts, modified a lot
def apply_gradients(policy, optimizer, grads_and_vars):
	actor_apply_ops = policy._actor_optimizer.apply_gradients(
        policy._actor_grads_and_vars)

	cgrads = policy._critic_grads_and_vars
	cutoff = len(cgrads) // (2 + ENSEMBLE_CNT)
	if policy.config["twin_q"]:
		critic_apply_ops = [
            policy._critic_optimizer[i].apply_gradients(cgrads[(i * cutoff):((i+1) * cutoff)]) \
                for i in range(2 + ENSEMBLE_CNT)
        ]
	else:
		critic_apply_ops = [
            policy._critic_optimizer[0].apply_gradients(cgrads)
        ]

	if policy.config["framework"] in ["tf2", "tfe"]:
		policy._alpha_optimizer.apply_gradients(policy._alpha_grads_and_vars)
		return
	else:
		alpha_apply_ops = policy._alpha_optimizer.apply_gradients(
			policy._alpha_grads_and_vars,
			global_step=tf1.train.get_or_create_global_step())
		return tf.group([actor_apply_ops, alpha_apply_ops] + critic_apply_ops)

# Same as above, also deleted a lot
def stats(policy, train_batch):
    return {
        # "policy_t": policy.policy_t,
        # "td_error": policy.td_error,
        "mean_td_error": tf.reduce_mean(policy.td_error),
        "actor_loss": tf.reduce_mean(policy.actor_loss),
        "critic_loss": tf.reduce_mean(policy.critic_loss),
        "alpha_loss": tf.reduce_mean(policy.alpha_loss),
        "alpha_value": tf.reduce_mean(policy.alpha_value),
        "target_entropy": tf.constant(policy.target_entropy),
        "mean_q": tf.reduce_mean(policy.q_t),
        "max_q": tf.reduce_max(policy.q_t),
        "min_q": tf.reduce_min(policy.q_t),
    }

# Same as above, also deleted a lot
class ActorEnsembleCriticOptimizerMixin(ActorCriticOptimizerMixin):
	def __init__(self, config):
        # - Create global step for counting the number of update operations.
        # - Use separate optimizers for actor & critic.
		ActorCriticOptimizerMixin.__init__(self, config)
		if config["framework"] in ["tf2", "tfe"]:
			assert False
		else:
			for _ in range(ENSEMBLE_CNT):
				self._critic_optimizer.append(
					tf1.train.AdamOptimizer(learning_rate=config[
						"optimization"]["critic_learning_rate"]))


def setup_early_mixins(policy, obs_space, action_space, config):
    ActorEnsembleCriticOptimizerMixin.__init__(policy, config)

ensembleQPolicy = SACTFPolicy.with_updates(
	name="ensembleQpolicy",
	make_model=build_ensembleQ_model,
	# TODO: add stats
	loss_fn=sac_actor_ensemble_critic_loss,
	gradients_fn=gradients_fn,
	apply_gradients_fn=apply_gradients,
    before_init=setup_early_mixins,
)
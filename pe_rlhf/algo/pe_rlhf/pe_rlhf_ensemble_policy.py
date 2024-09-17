from pe_rlhf.algo.pe_rlhf.pe_rlhf import PE_RLHFPolicy
from pe_rlhf.algo.ensembleQ.ensembleQ_model import ENSEMBLE_CNT
from pe_rlhf.algo.pe_rlhf.pe_rlhf_ensemble_model import build_ensembleQ_model, PE_RLHFEnsembleModel
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
tf, _, _ = try_import_tf()
tf1 = tf
tfp = try_import_tfp()
from ray.rllib.policy.sample_batch import SampleBatch
from pe_rlhf.algo.sac_lag.sac_lag_policy import get_dist_class, ActorCriticOptimizerMixin


# Redefined the loss function
NEWBIE_ACTION = "newbie_action"
TAKEOVER = "takeover"

# Changed a name, originally it was sac_actor_ensemble_critic_loss
def pe_rlhf_ensemble_ac_loss(policy, model: PE_RLHFEnsembleModel, _, train_batch):
	_ = train_batch[policy.config["info_total_cost_key"]]  # Touch this item, this is helpful in ray 1.2.0

	with tf.variable_scope('lambda'):
		param_init = 1e-8
		lambda_param = tf.get_variable(
			'lambda_value',
			initializer=float(param_init),
			trainable=False,
			dtype=tf.float32
		)
	policy.lambda_value = lambda_param


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
		raise ValueError("Doesn't support yet")
    # Continuous actions case.
	else:
        # Sample simgle actions from distribution.
		action_dist_class = get_dist_class(policy.config, policy.action_space)
		action_dist_t = action_dist_class(
			model.get_policy_output(model_out_t), policy.model)
		policy_t = action_dist_t.sample() if not deterministic else \
			action_dist_t.deterministic_sample()
		log_pis_t = tf.expand_dims(action_dist_t.logp(policy_t),
								   -1)  # Log probability of actions under current policy (policy_t)

		log_expert_a_t = action_dist_t.logp(train_batch[SampleBatch.ACTIONS])  # New compared to HACO
		log_agent_a_t = action_dist_t.logp(train_batch[NEWBIE_ACTION])  # New compared to HACO

		action_dist_tp1 = action_dist_class(
			model.get_policy_output(model_out_tp1), policy.model)
		policy_tp1 = action_dist_tp1.sample() if not deterministic else \
			action_dist_tp1.deterministic_sample()
		log_pis_tp1 = tf.expand_dims(action_dist_tp1.logp(policy_tp1), -1)

		# Q-values for the actually selected actions.
		q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
		if policy.config["twin_q"]:
			twin_q_t = model.get_twin_q_values(
				model_out_t, train_batch[SampleBatch.ACTIONS])
			# New code compared to HACO
			extra_q_ts = model.get_extra_q_values(
				model_out_t, train_batch[SampleBatch.ACTIONS])

		# Compared to the original, added cost
		# Cost Q-Value for actually selected actions
		c_q_t = model.get_cost_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
		if policy.config["twin_cost_q"]:
			twin_c_q_t = model.get_twin_cost_q_values(
				model_out_t, train_batch[SampleBatch.ACTIONS])

		q_t_det_policy = model.get_q_values(model_out_t, policy_t)
		if policy.config["twin_q"]:
			twin_q_t_det_policy = model.get_twin_q_values(
				model_out_t, policy_t)
			# New code
			extra_q_t_det_policy = model.get_extra_q_values(
				model_out_t, policy_t)
			q_t_det_policy = tf.reduce_min(
				(q_t_det_policy, twin_q_t_det_policy, *extra_q_t_det_policy),
				axis=0)  # Not sure if this is missing: extra_q_t_det_policy

		# Cost Q-values for current policy in given current state.
		c_q_t_det_policy = model.get_cost_q_values(model_out_t, policy_t)
		if policy.config["twin_cost_q"]:
			twin_c_q_t_det_policy = model.get_twin_cost_q_values(
				model_out_t, policy_t)
			c_q_t_det_policy = tf.reduce_min(
				(c_q_t_det_policy, twin_c_q_t_det_policy), axis=0)

		# Target Q network evaluation
		q_tp1 = policy.target_model.get_q_values(target_model_out_tp1, policy_tp1)
		if policy.config["twin_q"]:
			twin_q_tp1 = policy.target_model.get_twin_q_values(
				target_model_out_tp1, policy_tp1)
			# Compared to the original, missing:
			# extra_q_tp1 = policy.target_model.get_extra_q_values(
			#               target_model_out_tp1, policy_tp1)
			# Take min over both twin-NNs.
			# TODO: choose two random q nets from all ensembled nets and take min
			q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)

        # target c-q network evaluation
		c_q_tp1 = policy.target_model.get_cost_q_values(target_model_out_tp1,
														policy_tp1)
		if policy.config["twin_cost_q"]:
			twin_c_q_tp1 = policy.target_model.get_twin_cost_q_values(
				target_model_out_tp1, policy_tp1)
			# Take min over both twin-NNs.
			c_q_tp1 = tf.reduce_min((c_q_tp1, twin_c_q_tp1), axis=0)

		q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
		if policy.config["twin_q"]:
			twin_q_t_selected = tf.squeeze(twin_q_t, axis=len(twin_q_t.shape) - 1)
			# NEW code
			extra_qs_selected = [tf.squeeze(i, axis=len(q_t.shape) - 1) for i in extra_q_ts]

		# c_q_t selected
		c_q_t_selected = tf.squeeze(c_q_t, axis=len(c_q_t.shape) - 1)
		if policy.config["twin_cost_q"]:
			twin_c_q_t_selected = tf.squeeze(twin_c_q_t, axis=len(twin_c_q_t.shape) - 1)

		q_tp1 -= model.alpha * log_pis_tp1

		q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
		q_tp1_best_masked = (1.0 - tf.cast(train_batch[SampleBatch.DONES],
                                           tf.float32)) * q_tp1_best

	c_q_tp1_best = tf.squeeze(input=c_q_tp1, axis=len(c_q_tp1.shape) - 1)
	c_q_tp1_best_masked = \
		(1.0 - tf.cast(train_batch[SampleBatch.DONES], tf.float32)) * \
		c_q_tp1_best

	# compute RHS of bellman equation
	q_t_selected_target = tf.stop_gradient(
		train_batch[SampleBatch.REWARDS] +
		policy.config["gamma"] ** policy.config["n_step"] * q_tp1_best_masked)

	# Compute Cost of bellman equation.
	c_q_t_selected_target = tf.stop_gradient(train_batch[policy.config["info_cost_key"]] +
												policy.config["gamma"] ** policy.config["n_step"] * c_q_tp1_best_masked)

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

    # Compute the Cost TD-error (potentially clipped).
	base_c_td_error = tf.math.abs(c_q_t_selected - c_q_t_selected_target)
	if policy.config["twin_cost_q"]:
		twin_c_td_error = tf.math.abs(twin_c_q_t_selected - c_q_t_selected_target)
		c_td_error = 0.5 * (base_c_td_error + twin_c_td_error)
	else:
		c_td_error = base_c_td_error

	# conservative loss
	newbie_q_t = model.get_q_values(model_out_t, train_batch[NEWBIE_ACTION])
	if policy.config["twin_q"]:
		newbie_twin_q_t = model.get_twin_q_values(
			model_out_t, train_batch[NEWBIE_ACTION])
		# NEW code
		newbie_ensemble_q_t = model.get_extra_q_values(
			model_out_t, train_batch[NEWBIE_ACTION]
		)

	newbie_q_t_selected = tf.squeeze(newbie_q_t, axis=len(newbie_q_t.shape) - 1)
	if policy.config["twin_q"]:
		newbie_twin_q_t_selected = tf.squeeze(newbie_twin_q_t, axis=len(newbie_twin_q_t.shape) - 1)
		# NEW code
		newbie_ensemble_q_t_selected = [tf.squeeze(i, axis=len(i.shape)-1) for i in newbie_ensemble_q_t]

	# add conservative loss
	if policy.config["no_cql"]:
		critic_loss = [0.5 * tf.keras.losses.MSE(y_true=q_t_selected_target, y_pred=q_t_selected)]
	else:
		critic_loss = [
			0.5 * tf.keras.losses.MSE(
				y_true=q_t_selected_target, y_pred=q_t_selected) - tf.reduce_mean((tf.cast(train_batch[TAKEOVER],
																						tf.float32)) * policy.config[
																					"alpha"] * (
																						q_t_selected - newbie_q_t_selected))]
	if policy.config["twin_q"]:
		if policy.config["no_cql"]:
			loss = [0.5 * tf.keras.losses.MSE(y_true=q_t_selected_target, y_pred=twin_q_t_selected)]
			# NEW code
			for extra_q_t_selected in extra_qs_selected:
				loss.append(0.5 * tf.keras.losses.MSE(
					y_true=q_t_selected_target, y_pred=extra_q_t_selected
				))
		else:
			loss = [0.5 * tf.keras.losses.MSE(y_true=q_t_selected_target, y_pred=twin_q_t_selected) - \
					tf.reduce_mean((tf.cast(train_batch[TAKEOVER], tf.float32)) * policy.config["alpha"] * \
					(twin_q_t_selected - newbie_twin_q_t_selected))]
			# NEW code
			for (_q_t, _newbie_q_t) in zip(extra_qs_selected, newbie_ensemble_q_t_selected):
				loss.append(0.5 * tf.keras.losses.MSE(y_true=q_t_selected_target, y_pred=_q_t) - \
						tf.reduce_mean((tf.cast(train_batch[TAKEOVER], tf.float32)) * policy.config["alpha"] * \
						(_q_t - _newbie_q_t)))
		critic_loss.extend(loss)

	# add cost critic
	critic_loss.append(
		0.5 * tf.keras.losses.MSE(
			y_true=c_q_t_selected_target, y_pred=c_q_t_selected))
	if policy.config["twin_cost_q"]:
		critic_loss.append(0.5 * tf.keras.losses.MSE(
			y_true=c_q_t_selected_target, y_pred=twin_c_q_t_selected))

	# Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
	if model.discrete:
		raise ValueError("Didn't support discrete mode yet")
	else:
		alpha_loss = -tf.reduce_mean(
			model.log_alpha *
			tf.stop_gradient(log_pis_t + model.target_entropy))
		reward_loss = tf.reduce_mean(
			model.alpha * log_pis_t - q_t_det_policy)
		cost_loss = tf.reduce_mean(c_q_t_det_policy)
		actor_loss = tf.reduce_mean(
			model.alpha * log_pis_t - q_t_det_policy + c_q_t_det_policy)

    # add imitation loss to alpha loss
	# imitating both expert and agent itself
	self_regularization_loss = -policy.config["il_agent_coef"] * log_agent_a_t # NEW
	bc_loss = -policy.config["il_expert_coef"] * log_expert_a_t	# NEW
	# self_regularization_loss = - 0.05 * log_agent_a_t
	# print("Actor loss", actor_loss)
	# print("il loss", self_regularization_loss)

	# save for stats function
	policy.policy_t = policy_t
	policy.cost_loss = cost_loss
	policy.reward_loss = reward_loss
	policy.mean_batch_cost = train_batch[policy.config["info_cost_key"]]
	policy.q_t = q_t
	policy.c_q_tp1 = c_q_tp1
	policy.c_q_t = c_q_t
	policy.td_error = td_error
	policy.c_td_error = c_td_error
	policy.actor_loss = actor_loss + self_regularization_loss + bc_loss
	policy.critic_loss = critic_loss
	policy.c_td_target = c_q_t_selected_target
	policy.alpha_loss = alpha_loss
	policy.alpha_value = model.alpha
	policy.target_entropy = model.target_entropy
	policy.self_regularization_loss = self_regularization_loss
	policy.bc_loss = bc_loss

	# in a custom apply op we handle the losses separately, but return them
	# combined in one loss for now
	return actor_loss + tf.math.add_n(critic_loss) + alpha_loss


def gradients_fn(policy, optimizer, loss):
    # Eager: Use GradientTape.
	if policy.config["framework"] in ["tf2", "tfe"]:
		raise ValueError()
		tape = optimizer.tape
		pol_weights = policy.model.policy_variables()
		actor_grads_and_vars = list(
			zip(tape.gradient(policy.actor_loss, pol_weights), pol_weights))
		q_weights = policy.model.q_variables()
		c_q_weights = policy.model.cost_q_variables()
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

		if policy.config["twin_cost_q"]:
			c_half_cutoff = len(c_q_weights) // 2
			grads_3 = tape.gradient(policy.critic_loss[-2],
									c_q_weights[:c_half_cutoff])
			grads_4 = tape.gradient(policy.critic_loss[-1],
									c_q_weights[c_half_cutoff:])

			c_critic_grads_and_vars = \
				list(zip(grads_3, c_q_weights[:c_half_cutoff])) + \
				list(zip(grads_4, c_q_weights[c_half_cutoff:]))
		else:
			c_critic_grads_and_vars = list(zip(tape.gradient(policy.critic_loss[-1], c_q_weights), c_q_weights))

		alpha_vars = [policy.model.log_alpha]
		alpha_grads_and_vars = list(
			zip(tape.gradient(policy.alpha_loss, alpha_vars), alpha_vars))
    # Tf1.x: Use optimizer.compute_gradients()
	else:
		actor_grads_and_vars = policy._actor_optimizer.compute_gradients(
            policy.actor_loss, var_list=policy.model.policy_variables())

		q_weights = policy.model.q_variables()
		c_q_weights = policy.model.cost_q_variables()

		if policy.config["twin_q"]:
			cutoff = len(q_weights) // (2 + ENSEMBLE_CNT)
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

		# NEW code
		if policy.config["twin_cost_q"]:
			c_half_cutoff = len(c_q_weights) // 2
			base_c_q_optimizer, twin_c_q_optimizer = policy._critic_optimizer[-2:]
			c_critic_grads_and_vars = base_c_q_optimizer.compute_gradients(
				policy.critic_loss[-2], var_list=c_q_weights[:c_half_cutoff]
			) + twin_c_q_optimizer.compute_gradients(
				policy.critic_loss[-1], var_list=c_q_weights[c_half_cutoff:])
		else:
			c_critic_grads_and_vars = policy._critic_optimizer[
				-1].compute_gradients(
				policy.critic_loss[-1], var_list=c_q_weights)

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
    # for cost critic
	# NEW code
	policy._c_critic_grads_and_vars = [(clip_func(g), v)
										for (g, v) in c_critic_grads_and_vars
										if g is not None]

	policy._alpha_grads_and_vars = [(clip_func(g), v)
									for (g, v) in alpha_grads_and_vars
									if g is not None]

	grads_and_vars = (
			policy._actor_grads_and_vars + policy._critic_grads_and_vars + policy._c_critic_grads_and_vars +
			policy._alpha_grads_and_vars)
	return grads_and_vars


def apply_gradients(policy, optimizer, grads_and_vars):
    actor_apply_ops = policy._actor_optimizer.apply_gradients(policy._actor_grads_and_vars)

    cgrads = policy._critic_grads_and_vars
    c_cgrads = policy._c_critic_grads_and_vars  # NEW
    cutoff = len(cgrads) // (2 + ENSEMBLE_CNT)
    if policy.config["twin_q"]:
        critic_apply_ops = [
            policy._critic_optimizer[i].apply_gradients(cgrads[(i * cutoff):((i + 1) * cutoff)]) \
            for i in range(2 + ENSEMBLE_CNT)
        ]
    else:
        critic_apply_ops = [
            policy._critic_optimizer[0].apply_gradients(cgrads)]

    # NEW code
    if policy.config["twin_cost_q"]:
        c_half_cutoff = len(c_cgrads) // 2
        critic_apply_ops += [policy._critic_optimizer[-2].apply_gradients(c_cgrads[:c_half_cutoff]),
                             policy._critic_optimizer[-1].apply_gradients(c_cgrads[c_half_cutoff:])]
    else:
        critic_apply_ops.append(policy._critic_optimizer[-1].apply_gradients(c_cgrads))

    if policy.config["framework"] in ["tf2", "tfe"]:
        policy._alpha_optimizer.apply_gradients(policy._alpha_grads_and_vars)
        return
    else:
        alpha_apply_ops = policy._alpha_optimizer.apply_gradients(
            policy._alpha_grads_and_vars,
            global_step=tf1.train.get_or_create_global_step())

        return tf.group([actor_apply_ops, alpha_apply_ops] + critic_apply_ops)


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


PE_RLHFEnsemblePolicy = PE_RLHFPolicy.with_updates(
    name="PE_RLHFEnsemblePolicy",
	make_model=build_ensembleQ_model,
	gradients_fn=gradients_fn,
	apply_gradients_fn=apply_gradients,
	# mixins=[
	# # TargetNetworkMixin, ActorEnsembleCriticOptimizerMixin, ComputeTDErrorMixin, UpdatePenaltyMixin
	# TargetNetworkMixin, ActorEnsembleCriticOptimizerMixin, ComputeTDErrorMixin, ConditionalUpdatePenaltyMixin
	# ],
    before_init=setup_early_mixins,
	loss_fn=pe_rlhf_ensemble_ac_loss)
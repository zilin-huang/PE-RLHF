from ray.rllib.agents.sac.sac import SACTrainer
from pe_rlhf.algo.ensembleQ.ensembleQ_policy import ensembleQPolicy

def get_policy_class(config):
    return ensembleQPolicy

ensembleQTrainer = SACTrainer.with_updates(
	name="ensembleQ",
	default_policy=ensembleQPolicy,
	get_policy_class=get_policy_class,
)
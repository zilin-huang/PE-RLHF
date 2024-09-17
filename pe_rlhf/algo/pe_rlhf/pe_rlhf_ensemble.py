from pe_rlhf.algo.pe_rlhf.pe_rlhf import PE_RLHFTrainer
from pe_rlhf.algo.pe_rlhf.pe_rlhf_ensemble_policy import PE_RLHFEnsemblePolicy

def get_policy_class(config):
    return PE_RLHFEnsemblePolicy

PE_RLHFEnsembleTrainer = PE_RLHFTrainer.with_updates(
	name="PE_RLHFEnsembleTrainer",
	default_policy=PE_RLHFEnsemblePolicy,
	get_policy_class=get_policy_class,
)

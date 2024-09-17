import datetime
import os

from pe_rlhf.algo.pe_rlhf.pe_rlhf import PE_RLHFTrainer
from pe_rlhf.utils.callback import PE_RLHFCallbacks
from pe_rlhf.utils.pe_rlhf_human_in_the_loop_env import HumanInTheLoopEnv
from pe_rlhf.utils.train import train
from pe_rlhf.utils.train_utils import get_train_parser
from pe_rlhf.algo.pe_rlhf.pe_rlhf_ensemble import PE_RLHFEnsembleTrainer

def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = args.exp_name or "PE_RLHF_{}".format(get_time_str())
    stop = {"timesteps_total": 8_000}

    config = dict(
        env=HumanInTheLoopEnv,
        env_config={
            "manual_control": True,
            "use_render": True,
            # "controller": "keyboard",  # use keyboard or not
            "window_size": (1600, 1100),
            "cos_similarity": True,

            # Can revise these parameters
            "ensemble": True,
            "warmup_ts": 1000,
            "value_takeover_threshold": 1.5,
            # "value_from_scratch": True,
            # "update_value_freq": 1000,

            "exp_path": os.path.join(args.local_dir, exp_name),  # Add experiment path
        },

        # ===== Training =====
        takeover_data_discard=False,
        twin_cost_q=True,
        alpha=10,
        no_reward=True,  # need reward
        explore=True,

        optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
        prioritized_replay=False,
        horizon=1000,
        target_network_update_freq=1,
        timesteps_per_iteration=100,
        metrics_smoothing_episodes=10,
        learning_starts=100,
        clip_actions=False,
        train_batch_size=1024,

        normalize_actions=True,
        num_cpus_for_driver=0.5,
        # No extra worker used for learning. But this config impacts the evaluation workers.
        num_cpus_per_worker=0.1,
        # num_gpus_per_worker=0.1 if args.num_gpus != 0 else 0,
        num_gpus=0.2 if args.num_gpus != 0 else 0,
    )

    train(
        PE_RLHFTrainer if not args.pe_rlhf_ensemble else PE_RLHFEnsembleTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=None,
        checkpoint_freq=args.ckpt_freq,  # Add checkpoint frequency
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=2,
        num_seeds=1,
        custom_callback=PE_RLHFCallbacks,
        # test_mode=True,
        # local_mode=True
        local_dir=args.local_dir,  # Add local directory
    )

# Trustworthy Human-AI Collaboration ✨: Reinforcement Learning with Human Feedback and Physics Knowledge for Safe Autonomous Driving



## 🚀 Getting Started
### 1. Requirements 📦

For an optimal experience, we recommend using conda to set up a new environment for PE-RLHF.

```bash
# Clone the code to local
git clone https://github.com/zilin-huang/PE-RLHF.git
cd PE-RLHF

# Create virtual environment
conda create -n PE-RLHF python=3.7
conda activate PE-RLHF

# Install basic dependency
pip install -e .

conda install cudatoolkit=11.0
conda install -c nvidia cudnn
# Now you can run the training script of PE-RLHF in MetaDrive Environment.
```

## 4. Training baselines
### RL and safe RL baselines 
For SAC/PPO/PPO-Lag/SAC-Lag, there is no additional requirement to run the training scripts. 

```bash
# use previous HAIM-DRL environment
conda activate PE-RLHF  
cd pe_rlhf/run_baselines
# launch baseline experiment
python train_[ppo/sac/sac_lag/ppo_lag].py --num-gpus=[your_gpu_num]
```

For example:
```bash
python train_ppo.py --num-gpus=1
```

📝 **Note:** The result reported in our paper for RL and safe RL methods were repeated five times using different random seeds. To save computer resource, can revise the `num_seeds=5` to `num_seeds=1` in the `train_[ppo/sac/sac_lag/ppo_lag].py`.

Then, the training and testing results for RL and safe RL methods will be saved in the run_baselines/[ppo/sac/sac_lag/ppo_lag] folder. You can open it with:

```bash
tensorboard --logdir=. --port=8080
```

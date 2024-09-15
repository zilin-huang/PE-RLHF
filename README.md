# Trustworthy Human-AI Collaboration ✨: Reinforcement Learning with Human Feedback and Physics Knowledge for Safe Autonomous Driving



## 1. Requirements 📦

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

## 2. Getting Started 🚀 


## 4. Training Baselines
### 4.1 Physics-based Methods 
The physics-based methods (e.g., IDM-MOBIL model) does not need to train a model, to eval the performance you can directly run:

```bash
# use previous PE-RLHF environment
conda activate PE-RLHF  
cd pe_rlhf/run_baselines
# eval physics-based experiment
python eval_IDM_MOBIL.py
```
Then, testing results for physics-based methods will be saved in the `idm_mobil_results.csv` file.

### 4.2 RL and Safe RL Methods 
For SAC/PPO/PPO-Lag/SAC-Lag, there is no additional requirement to run the training scripts. 

```bash
# use previous PE-RLHF environment
conda activate PE-RLHF  
cd pe_rlhf/run_baselines
# launch RL and safe RL experiment
python train_[ppo/sac/sac_lag/ppo_lag].py --num-gpus=[your_gpu_num]
```

For example:
```bash
python train_ppo.py --num-gpus=1
```

📝 **Note:** The result reported in our paper for RL and safe RL methods were repeated five times using different random seeds. To save computer resource, can revise the `num_seeds=5` to `num_seeds=1` in the `train_[ppo/sac/sac_lag/ppo_lag].py`.

Then, the training and testing results for RL and safe RL methods will be saved in the run_baselines/[ppo/sac/sac_lag/ppo_lag] folder. You can open it with tensorboard. 

For example:
```bash
tensorboard --logdir=. --port=8080
```

### 4.3 Human Demonstration Dataset
Human demonstration dataset is required to run offline RL, IL, and offline RLHF methods. You can collect human demonstration by runing:

```bash
# use previous PE-RLHF environment
conda activate PE-RLHF 
cd pe_rlhf/utils
# launch human demonstration data collection experiment
python collect_human_data_set.py
```
📝 **Note:** Also, for your convenient, we provide a high-quality demonstration dataset collected by our PE-RLHF team human expert. This dataset contains approximately 49,000 transitions in the training environment. This high-quality demonstration dataset achieves 100% success rate, with an episodic return of 388.16 and a low safety violation rate of 0.03. You can dricetly download at [here](https://github.com/zilin-huang/PE-RLHF/releases/tag/v1.0.0).

Next, move the dataset, for example, `human_traj_100_new.json`, from the Downloads directory to the 'pe_rlhf' directory (which is at the same level as the `algo`, `run_baselines`, `run_main_exp`, and `utils`. 

For example, you can use the following command:
```bash
mv ~/Downloads/human_traj_100_new.json /home/codes/PE-RLHF/pe_rlhf/
```

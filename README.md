# Trustworthy Human-AI Collaboration ✨: Reinforcement Learning with Human Feedback and Physics Knowledge for Safe Autonomous Driving

[**Webpage**](https://zilin-huang.github.io/PE-RLHF-website/) | 
[**Code**](https://github.com/zilin-huang/PE-RLHF) | 
[**Video**](https://www.youtube.com/embed/7TQ5ZCrqtfI?si=r6HAMMDa3XVdAITO) |
[**Paper**](https://arxiv.org/abs/2409.00858) 

[<video width="600" controls>
  <source src="https://github.com/zilin-huang/PE-RLHF-website/blob/master/static/videos/RLHF.mp4" type="video/mp4">
</video>](https://github.com/zilin-huang/PE-RLHF-website/blob/master/static/videos/RLHF.mp4)


## 🌟 Highlights
- **`2024-09-16`** **Codes are now release!** 
- **`2024-09-12`** Explore our project page, now live [here](https://zilin-huang.github.io/PE-RLHF-website/)🔗!
- **`2024-09-01`** Our paper is available on [Arxiv](https://arxiv.org/abs/2409.00858)📄!


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

## 2. Quick Start 🚀 
Since the main experiment of PE-RLHF takes one hour and requires a steering wheel (Logitech G920), we further provide an 
easy task for users to experience PE-RLHF.

```bash
# use PE-RLHF environment
conda activate PE-RLHF 
cd pe_rlhf/run_main_exp/
python train_pe_rlhf_keyboard_easy.py --exp-name PE_RLHF --local-dir PATH_TO_SAVE_DIR --num-gpus=[your_gpu_num]
```
For example:
```bash
python train_pe_rlhf_keyboard_easy.py --exp-name PE_RLHF --local-dir /home/sky-lab/codes/PE-RLHF/pe_rlhf/run_main_exp --num-gpus=1
```

Now, human is authorized to take over the vehicle by pressing **W/A/S/D** and guide or safeguard the agent to the destination ("E" can be used to pause simulation).  Since there is only one map in this task, 10 minutes or 5000 transitions is enough for agent to learn a policy.

📝 **Note:** To make it easier for you to get started and test our code, we have uploaded a pretrained value estimator, i.e., `sac_expert.npz`, in the `utils/saved_expert folder`. Meanwhile, we set `"warmup_ts": 0`, so the info interface will show warmup: False. i.e., no need to warm up.


## 3. Main Experiment
To reproduce the main experiment reported in paper, run following scripts:
```bash
python train_pe_rlhf.py --exp-name PE_RLHF --local-dir PATH_TO_SAVE_DIR --num-gpus=[your_gpu_num]
```
If steering wheel is not available, set ```controller="keyboard"``` in the script to train HAIM-DRL agent. After launching this script,
one hour is required for human to assist HAIM-DRL agent to learn a generalizable driving policy by training in 50 different maps.

📊 **Visualizing Results:** Then, the training results will be saved in the run_main_exp/PE_RLHF folder. To evaluate PE-RLHF, you should run `run_main_exp/eval_pe_rlhf.py`.

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
📊 **Visualizing Results:** Then, testing results for physics-based methods will be saved in the `idm_mobil_results.csv` file.

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

📊 **Visualizing Results:** Then, the training and testing results for RL and safe RL methods will be saved in the run_baselines/[ppo/sac/sac_lag/ppo_lag] folder. You can open it with tensorboard. 

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
mv ~/Downloads/human_traj_100_new.json /home/sky-lab/codes/PE-RLHF/pe_rlhf/
```

### 4.4 IL Methods
If you wish to run BC/CQL, extra setting is required as follows:
```bash
# ray needs to be updated to 1.2.0
pip install ray==1.2.0
cd pe_rlhf/run_baselines
# launch BC/CQL experiment
python train_[bc/cql].py --num-gpus=0 # do not use gpu
```
⚙️ **Issue:** BC/CQL will encounter the following error:

```bash
File "/home/zilin/anaconda3/envs/PE-RLHF/lib/python3.7/site-packages/ray/rllib/utils/torch_ops.py", line 105, in mapping
    tensor = torch.from_numpy(np.asarray(item))
TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
```
💡 **Solution:** Modify lines 104-105 of the original code, i,e., `tensor = torch.from_numpy(np.asarray(item))`. The new code as
```bash
else:
    # tensor = torch.from_numpy(np.asarray(item))
    if isinstance(item, bool):
        item = int(item)
    tensor = torch.from_numpy(np.asarray(item).astype(float))
```

📝 **Note:** Since the computational process of BC and CQL is on the CPU, it requires a relatively large amount of CPU memory. To reduce the computational resources, you can revise the `num_seeds=5` to `num_seeds=1` in the `train_[bc/cql].py`. Also we set `bc_iters=tune.grid_search([5_0000, 10_0000]` in `train_cql.py`. You can also change it to `bc_iters=5_0000`. this will also save computational resources.

### 4.5 GAIL and Offline RLHF Methods
To run GAIL/HG-DAgger/IWR, please create a new conda environment and install GPU-version of torch:
```bash
# Create virtual environment
conda create -n PE-RLHF-torch python=3.7
conda activate PE-RLHF-torch

# Install basic dependency
pip install -e .

# install torch
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install cudatoolkit=11.0
```
Now, GAIL can be trained by:
```bash
cd pe_rlhf/run_baselines 
python train_[gail/IWR/hg_dagger].py
```
📝 **Note:** You can set the maximum number of training cycles in `train_gail.py` by modifying `parser.add_argument('--max_epoch', default=3000)`. 


IWR/HG-Dagger can be trained by:
```bash
cd pe_rlhf/run_baselines 
python train_[IWR/hg_dagger].py
```

📝 **Note:** IWR and HG-Dagger run through a warm-up period of renderless, then the metadrive render screen pops up, with the human-in-the-loop capability activated to allow a human takeover. You can set the number of training rounds by modifying `NUM_ITS = 5` in `train_[IWR/hg_dagger].py`. Also, you can adjust `BC_WARMUP_DATA_USAGE = 30000` in `train_[IWR/hg_dagger].py` to set a different number of warmup transitions. 

📊 **Visualizing Results:** Then, the training model for IWR and HG-Dagger will be saved in the run_baselines folder. Notably, to evaluate IWR and HG-Dagger, you should run `algo/IWR/eval_IWR.py` or `algo/HG_Dagger/eval_hg_dagger.py`.

### 4.6 Online RLHF Methods
If you want to run HACO

```bash
cd pe_rlhf/run_baselines 
python train_haco.py --num-gpus=1
```

📝 **Note:** If steering wheel is not available, uncomment line 25 in `train_haco.py`, i.e., `# “controller”: “keyboard”, # use keyboard or not`. This allows you to use the keyboard to control agent.

📊 **Visualizing Results:** Then, the training results will be saved in the run_baselines/HACO folder. To evaluate HACO, you should run `run_baselines/eval_haco.py`.



## 5. Ablation Study
The `pe_rlhf_human_in_the_loop_env.py` is the core file to implement the PE-RLHF. You can examine the sensitivity of PE-RLHF as well as perform ablation experiments by adjusting the settings in `lines 52 - 67`. In addition, you can also do this by adjusting `lines 30 - 35` of `train_pe_rlhf.py`.

You can also achieve this effect by setting command line parameters. For example:
```bash
python train_pe_rlhf.py --exp-name PE_RLHF --local-dir /home/sky-lab/codes/PE-RLHF/pe_rlhf/run_main_exp --num-gpus=1 --pe_rlhf-ensemble --ckpt-freq 10
```

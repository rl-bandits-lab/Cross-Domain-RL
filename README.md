# QAvatar: Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics

This repository is the official implementation of [Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics]

## Requirements package:
```
gym version: 0.21.0
mujoco-py version: 2.1.2.14
robosuite version: 1.3.0
numpy version: 1.24.3
torch version: 2.2.1
```

## Training
Step 1: Train the Source Model and Collect Source-Domain Data.
Run the following commands to train the source model and collect the corresponding source-domain data:

```
cd {Absolute path to QAvatar}source/stable-baselines3-1.7.0/

python3 train_src_model.py \
    --seed 1 \
    --device 'cuda:0' \
    --folder {Absolute path to the folder for saving the source model} \
    --env 'HalfCheetah-v3'

python3 data_collect.py \
    --seed 1 \
    --folder {Absolute path to the folder for saving the source model} \
    --env 'HalfCheetah-v3'
```

Step 2: Train the Normalizing Flow Model.
Run the following commands to train the normalizing flow model described in the paper:

```
cd {Absolute path to QAvatar}source/flowpg
**If training the joint flow model (i.e., modeling the joint distribution of (s, a) pairs), run:**

python -m experiments.train_joint_flow \
    --seed 1 \
    --device 'cuda:0' \
    --data_folder {Absolute path to the folder containing the collected data}

If training the independent flow models (i.e., modeling states and actions independently), run:

python -m experiments.train_state_flow \
    --seed 1 \
    --device 'cuda:0' \
    --data_folder {Absolute path to the folder containing the collected data}

python -m experiments.train_action_flow \
    --seed 1 \
    --device 'cuda:0' \
    --data_folder {Absolute path to the folder containing the collected data}
    
```

Step 3: Train the Target-Domain Model.
Run the following command to train the target-domain model as described in the paper:
```
cd {Absolute path to QAvatar}target_domain/

python3 train_target_model.py \
    --seed 1 \
    --device 'cuda:0' \
    --xml_file cheetah_target.xml \
    --src_env 'HalfCheetah-v3' \
    --env 'HalfCheetah-v3' \
    --src_folder {Absolute path to the folder containing the source model} \
    --flow_folder {Absolute path to the folder containing the flow model} \
    --tar_folder {Absolute path to the folder for saving the target model} \
    --joint_flow {True or False depending on whether to use the joint flow model}

```

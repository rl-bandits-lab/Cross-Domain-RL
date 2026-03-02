import os
import argparse
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import SAC
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import gym
import numpy as np

from train_src_model import construct_env
from robosuite.wrappers import GymWrapper
import robosuite as suite

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def put_transition(buffer, *transition):
    buffer.append(transition)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, nargs='?', default=1)
parser.add_argument("--folder", type=str, nargs='?', default='/home/')
parser.add_argument("--env", type=str, nargs='?', default='HalfCheetah-v3')
parser.add_argument("--device", type=str, nargs='?', default='cuda:0')
args = parser.parse_args()

log_folder = f"{args.folder}logs/source/{args.env}/seed_{str(args.seed)}/"

model = SAC.load(log_folder + "best_model", device = args.device)
data_save_folder = log_folder + 'corresponding_data_collection/'
os.makedirs(data_save_folder, exist_ok=True)

vec_env, _, _, _ = construct_env(args.env)
vec_env.seed(seed=args.seed)
set_random_seed(seed = args.seed)

state_list = []
action_list = []
sample_num = 60000

for i in range(sample_num):
    obs = vec_env.reset()
    dones = False
    reward = 0
    
    while not dones:
        action, _ = model.predict(obs)
        state_list.append(obs[0])
        action_list.append(action[0])
        obs_next, rewards, dones, info = vec_env.step(action)

        obs = obs_next
        reward += rewards
        if(len(action_list) >= sample_num):
            break
    print(f'episode: {i}, reward: {reward}')
    if(len(action_list) >= sample_num):
        break

np.save(data_save_folder + 'state.npy', np.array(state_list))
np.save(data_save_folder + 'action.npy', np.array(action_list))
print("\nThe data is saved in the folder:", data_save_folder)
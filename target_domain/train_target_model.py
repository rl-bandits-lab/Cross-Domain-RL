import os
import gym
import argparse
import torch as th
import torch.nn as nn
import numpy as np
import os
import random
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.sac.sac import SAC, SAC1, SAC2
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from core.flow.real_nvp import RealNvp

from robosuite.wrappers import GymWrapper
import robosuite as suite

def save_and_print_config(args, gamma, target_log_folder):
    print("\nConfig, target model, and results saved in folder:", target_log_folder)
    print("args: ", args)
    print("")
    with open(target_log_folder + "config.txt", mode='w') as f:
        f.write(f"decoder_lr: {args.decoder_lr}\n")
        f.write(f"target_env: {args.env}\n")
        f.write(f"xml_file: {args.xml_file}\n")
        f.write(f"src_env: {args.src_env}\n")
        f.write(f"flow_folder: {args.flow_folder}\n")
        f.write(f"joint_flow: {args.joint_flow}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"src_folder: {args.src_folder}\n")
        f.write(f"gamma: {gamma}\n")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

class decoder_network(nn.Module):
    def __init__(self, source_state_dim, source_action_dim, target_state_dim, target_action_dim, device, useTanh, hidden_size=256):
        super(decoder_network, self).__init__()
        self.state_emb = nn.Sequential(
            nn.Linear(target_state_dim, hidden_size//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.LeakyReLU(0.2),
        )
        self.action_emb = nn.Sequential(
            nn.Linear(target_action_dim, hidden_size//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.LeakyReLU(0.2),
        )

        self.out_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, source_state_dim + source_action_dim),
        )
        self.useTanh = useTanh
        self.device = device
        self.source_state_dim = source_state_dim
        self.source_action_dim = source_action_dim

    def forward(self, target_state, target_action):
        state_emb = self.state_emb(target_state.float())
        action_emb = self.action_emb(target_action.float())
        input = th.cat((state_emb, action_emb), dim=1)
        output = self.out_layer(input)
        if self.useTanh:
            output = nn.Tanh()(output)
        return output

def construct_env(src_env_name, tar_env_name, xml_file=None):
    if xml_file is not None:
        print(f"Construct environment with source domain '{src_env_name}' and target domain '{tar_env_name}' using XML file '{xml_file}'.")
        source_env = DummyVecEnv([lambda: gym.make(src_env_name)]*1)
        target_env = DummyVecEnv([lambda: gym.make(tar_env_name, xml_file = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/" + xml_file)))]*1)
        eval_env = DummyVecEnv([lambda: Monitor(gym.make(tar_env_name, xml_file = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/"+ xml_file))))]*1)
        env_gamma = 0.99    
    elif tar_env_name == 'Door' or tar_env_name == 'Lift' or tar_env_name == 'Wipe':
        print(f"Construct robosuite environment with source domain '{src_env_name}' using Panda robots and target domain '{tar_env_name}' using UR5e robots.")
        source_env = DummyVecEnv([lambda: GymWrapper(
            suite.make(
                src_env_name,
                robots="Panda", 
                use_camera_obs=False, 
                has_offscreen_renderer=False, 
                has_renderer=False,  
                reward_shaping=True, 
                control_freq=20, 
            )
        )]*1)
        target_env = DummyVecEnv([lambda: GymWrapper(
            suite.make(
                tar_env_name,
                robots="UR5e", 
                use_camera_obs=False,
                has_offscreen_renderer=False, 
                has_renderer=False,
                reward_shaping=True, 
                control_freq=20, 
            )
        )]*1)
        eval_env = DummyVecEnv([lambda: Monitor(GymWrapper(
            suite.make(
                tar_env_name,
                robots="UR5e", 
                use_camera_obs=False,  
                has_offscreen_renderer=False, 
                has_renderer=False,  
                reward_shaping=True, 
                control_freq=20,
            )
        ))]*1)
        env_gamma = 0.9
    else:
        print(f"Construct robosuite environment with source domain '{src_env_name}' and target domain '{tar_env_name}'.")
        source_env = DummyVecEnv([lambda: gym.make(src_env_name)]*1)
        target_env = DummyVecEnv([lambda: gym.make(tar_env_name)]*1)
        eval_env = DummyVecEnv([lambda: Monitor(gym.make(tar_env_name))]*1)
        env_gamma = 0.99

    return source_env, target_env, eval_env, env_gamma
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs='?', default=2)
    parser.add_argument("--decoder_lr", type=float, nargs='?', default=1e-4)
    parser.add_argument("--device", type=str, nargs='?', default="cuda:3")
    parser.add_argument("--src_folder", type=str, nargs='?', default='/home/')
    parser.add_argument("--flow_folder", type=str, nargs='?', default='/home/')
    parser.add_argument("--tar_folder", type=str, nargs='?', default='/home/')
    parser.add_argument("--xml_file", default=None, type=str) 
    parser.add_argument("--env", type=str, nargs='?', default='HalfCheetah-v3')
    parser.add_argument("--src_env", type=str, nargs='?', default='HalfCheetah-v3')
    parser.add_argument("--joint_flow", type=str, default='True') 
    args = parser.parse_args()

    seed_everything(args.seed)
    target_log_folder = f"{args.tar_folder}{args.env}/seed_{str(args.seed)}/"

    source_env, target_env, eval_env, gamma = construct_env(args.src_env, args.env, args.xml_file)

    target_env.seed(seed=args.seed)
    eval_env.seed(seed=args.seed)
    set_random_seed(seed = args.seed)
    
    args.flow_folder += f"{str(args.seed)}/"
    args.src_folder += f"{str(args.seed)}/"

    new_logger = configure(target_log_folder, ["stdout", "csv", "tensorboard"])
    eval_callback = EvalCallback(eval_env, best_model_save_path=target_log_folder, log_path=target_log_folder, eval_freq=2000, deterministic=True, render=False, n_eval_episodes = 5)
    save_and_print_config(args, gamma, target_log_folder)

    source_model = SAC.load(args.src_folder + "best_model", device = args.device)
    decoder = decoder_network(source_env.observation_space.shape[0], source_env.action_space.shape[0], target_env.observation_space.shape[0],target_env.action_space.shape[0], args.device, True).to(args.device)

    if args.joint_flow == 'True':
        flow_model = RealNvp.load_module(args.flow_folder + "joint/flow.pt").to(args.device)
        src_std = th.tensor(np.load(args.flow_folder + "joint/src_info.npz")['std']).to(args.device)
        src_mean = th.tensor(np.load(args.flow_folder + "joint/src_info.npz")['mean']).to(args.device)

        target_model = SAC1("MlpPolicy", env = target_env, SACmodel = source_model, decoder = decoder, flow_model = flow_model,
                            decoder_lr = args.decoder_lr,src_std = src_std, 
                            src_mean = src_mean, verbose = 1, device = args.device, seed = args.seed, gamma = gamma)
    elif args.joint_flow == 'False':
        state_flow_model = RealNvp.load_module(args.flow_folder + "state/flow.pt").to(args.device)
        src_std = th.tensor(np.load(args.flow_folder + "state/src_info.npz")['std']).to(args.device)
        src_mean = th.tensor(np.load(args.flow_folder + "state/src_info.npz")['mean']).to(args.device)
        action_flow_model = RealNvp.load_module(args.flow_folder + "action/flow.pt").to(args.device)

        target_model = SAC2("MlpPolicy", env = target_env, SACmodel = source_model, decoder = decoder, state_flow = state_flow_model,
                            action_flow = action_flow_model, decoder_lr = args.decoder_lr,src_state_std = src_std,
                            src_state_mean = src_mean, verbose = 1, device = args.device, seed = args.seed, gamma = gamma)
    else:
        raise ValueError("Error: unknown argument args.joint_flow")

    target_model.set_logger(new_logger)
    target_model.learn(total_timesteps=int(5e5), progress_bar = True, callback=eval_callback)

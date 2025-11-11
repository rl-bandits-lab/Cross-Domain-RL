import os
import gym
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from robosuite.wrappers import GymWrapper
import robosuite as suite

def construct_env(src_env_name):
    if src_env_name == 'Door' or src_env_name == 'Lift' or src_env_name == 'Wipe':
        print(f"Construct robosuite environment with source domain '{src_env_name}' using Panda robot.")
        vec_env = DummyVecEnv([lambda: GymWrapper(
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
        eval_env = DummyVecEnv([lambda: Monitor(GymWrapper(
            suite.make(
                src_env_name,
                robots="Panda", 
                use_camera_obs=False, 
                has_offscreen_renderer=False, 
                has_renderer=False,  
                reward_shaping=True, 
                control_freq=20, 
            ))
        )]*1)
        env_gamma = 0.9
        timesteps = int(5e5)
    else:
        print(f"Construct environment with source domain '{src_env_name}'.")
        vec_env = DummyVecEnv([lambda: gym.make(src_env_name)]*1)
        eval_env = DummyVecEnv([lambda: Monitor(gym.make(src_env_name))]*1)
        env_gamma = 0.99
        timesteps = int(1e6)
    return vec_env, eval_env, env_gamma, timesteps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs='?', default=1)
    parser.add_argument("--device", type=str, nargs='?', default='cuda:0')
    parser.add_argument("--folder", type=str, nargs='?', default='')
    parser.add_argument("--env", type=str, nargs='?', default='HalfCheetah-v3')
    args = parser.parse_args()

    log_folder = f"{args.folder}logs/source/{args.env}/seed_{str(args.seed)}/"
    print(f"Source model has been saved in: {log_folder}")
    vec_env, eval_env, env_gamma, timesteps = construct_env(args.env)
    vec_env.seed(seed=args.seed)
    eval_env.seed(seed=args.seed)
    set_random_seed(seed = args.seed)

    new_logger = configure(log_folder, ["stdout", "csv", "tensorboard"])
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_folder, log_path=log_folder, eval_freq=2000, deterministic=True, render=False, n_eval_episodes = 5)

    model = SAC("MlpPolicy", vec_env, verbose = 1, device = args.device, seed = args.seed, gamma = env_gamma)
    model.set_logger(new_logger)
    model.learn(total_timesteps=timesteps, progress_bar = True, callback=eval_callback)
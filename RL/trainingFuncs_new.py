'''
Author: Qin Yusen
email: qinys2001@163.com
LastEditTime: 2025-10-27 20:26:03
Description: 
'''
import torch
import numpy as np
import os
import datetime
import math

import stable_baselines3 as SB3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gymnasium import spaces
from torch_geometric.data import Data
from typing import List, Dict, Any, Tuple, Optional

from stateGCNandSeqEncode import RLStateExtractor
from RLEnv import RLEnvironment
from xgboost import Booster

class PyGVecEnvWrapper(VecEnvWrapper):
    """
    A VecEnv wrapper that handles stacking for dictionary observations
    containing PyTorch Geometric Data objects.

    It stacks compatible spaces (like Box) using np.stack, but keeps
    the specified PyG data key as a list of Data objects.
    """
    def __init__(self, venv: SubprocVecEnv, pyg_data_key: str = 'graphData'):
        super().__init__(venv)
        self.pyg_data_key = pyg_data_key
        assert isinstance(self.observation_space, spaces.Dict), "This wrapper only works with Dict observation spaces"
        assert self.pyg_data_key in self.observation_space.spaces, f"Key '{self.pyg_data_key}' not found in observation space"

    def _process_obs_list(self, obs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not obs_list:
            return {key: np.array([]) for key in self.observation_space.spaces.keys()}

        stacked_obs = {}
        for key, subspace in self.observation_space.spaces.items():
            if key != self.pyg_data_key:
                try:
                    stacked_obs[key] = np.stack([single_obs[key] for single_obs in obs_list])
                except ValueError as e:
                    print(f"Warning: Failed to stack key '{key}'. Error: {e}. Returning as list.")
                    stacked_obs[key] = [single_obs[key] for single_obs in obs_list]
            else:
                 pyg_list = [single_obs[key] for single_obs in obs_list]
                 if not all(isinstance(item, Data) for item in pyg_list if item is not None):
                     print(f"Warning: Items for key '{self.pyg_data_key}' might not all be PyG Data objects.")
                 stacked_obs[self.pyg_data_key] = pyg_list

        return stacked_obs

    def reset(self) -> Dict[str, Any]:
        #obs_list = self.venv.reset() 
        for env_idx, remote in enumerate(self.venv.remotes):
            remote.send(("reset", (self.venv._seeds[env_idx], self.venv._options[env_idx])))
        results = [remote.recv() for remote in self.venv.remotes]
        obs, self.venv.reset_infos = zip(*results)  # type: ignore[assignment]
        # Seeds and options are only used once
        self.venv._reset_seeds()
        self.venv._reset_options()
        return self._process_obs_list(obs)

    def step_wait(self) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        #obs_list, rewards, dones, infos = self.venv.step_wait()
        results = [remote.recv() for remote in self.venv.remotes]
        self.venv.waiting = False
        obs, rews, dones, infos, self.venv.reset_infos = zip(*results)  # type: ignore[assignment]
        stacked_obs = self._process_obs_list(obs)
        return stacked_obs, rews, dones, infos

    # def step_async(self, actions: np.ndarray) -> None:
    #    self.venv.step_async(actions)

def make_envvec(design_lists, cut_weight_lists, device, rwd='shaped', seed=37, train_num=16, test_num=4):
    env_factory = [lambda i=count: Monitor(RLEnvironment(design_lists[i], device, cut_weight= cut_weight_lists[i], \
                     rwd=rwd)) for count in range(train_num)]
    train_env_vec = SubprocVecEnv(env_factory)
    train_env_vec = PyGVecEnvWrapper(train_env_vec)
    train_env_vec.seed(seed)
    env_factory = [lambda i=count: Monitor(RLEnvironment(design_lists[i], device, cut_weight= cut_weight_lists[i],\
                    rwd=rwd)) for count in range(train_num, train_num + test_num)]
    test_env_vec = SubprocVecEnv(env_factory)
    test_env_vec = PyGVecEnvWrapper(test_env_vec)
    test_env_vec.seed(seed-1)
    return train_env_vec, test_env_vec

def make_policy():
    optimizer = torch.optim.Adam
    policy_kwargs = dict(
        features_extractor_class = RLStateExtractor,
        features_extractor_kwargs={},
        net_arch = dict(pi=[256,128,32],vf=[256,128,32]),
        activation_fn = torch.nn.LeakyReLU,
        optimizer_class = optimizer
    )
    return policy_kwargs

def linear_schedule(initial_value, final_value = 0.0):
    if not (0.0 <= final_value <= initial_value):
        raise ValueError(
            f"Final value {final_value} must be between 0.0 "
            f"and initial_value {initial_value} (inclusive)."
        )

    def func(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)

    return func

def cosine_schedule(initial_value: float, final_value: float = 0.0):
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * (1 + math.cos(math.pi * (1 - progress_remaining))) / 2
    return func

def cal_psm_metric(actions_x, actions_y, gamma, eps=1e-8):
    abs_diff = torch.abs(torch.unsqueeze(actions_x,1) - torch.unsqueeze(actions_y,0))
    cost_mat = 0.5 * torch.sum(abs_diff, dim=-1)

def train_process(design_lists, cut_weight_lists, device, rwd = 'shaped', optim_lr = 6e-4):
    train_envs, test_envs = make_envvec(design_lists, cut_weight_lists, device, rwd)
    policy_kw = make_policy()
    logdir = os.path.join('log_training', datetime.datetime.now().strftime("%Y%m%d"))
    checkdir = os.path.join('checkpoints', datetime.datetime.now().strftime("%Y%m%d"))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(checkdir):
        os.makedirs(checkdir)
    model = PPO(
        policy = 'MultiInputPolicy',
        env = train_envs,
        learning_rate= cosine_schedule(optim_lr),
        policy_kwargs= policy_kw,
        verbose = 1,
        n_steps= 300,
        batch_size=256,
        n_epochs=8,
        device=device,
        tensorboard_log=logdir,
        ent_coef=0.2,
        clip_range=0.2,
        gamma=0.95,
        vf_coef=0.1,
    )
    eval_cback = EvalCallback(test_envs, eval_freq=1200,
                n_eval_episodes=20,
                best_model_save_path=checkdir+'best/')
    check_cback = CheckpointCallback(
        save_freq=300, save_path=checkdir, 
        name_prefix="model_train_point"
    )

    model.learn(total_timesteps=1152000, callback=[eval_cback, check_cback])
    train_envs.close()
    test_envs.close()

def continue_train_process(checkpoint_path: str, design_lists, cut_weight_lists, device, rwd='shaped', optim_lr=1e-4, more_timesteps=500000):
    """
    从一个保存的 checkpoint 恢复训练。

    :param checkpoint_path: 要加载的模型的 .zip 文件路径。
    :param design_lists: 用于环境的设计列表。
    :param cut_weight_lists: 用于环境的切割权重列表。
    :param device: 用于训练的设备 (例如, 'cuda' 或 'cpu')。
    :param rwd: 要使用的奖励方案 ('shaped' 或 'native')。
    :param optim_lr: 新训练阶段的初始学习率。
    :param more_timesteps: 额外要训练的时间步数。
    """
    train_envs, test_envs = make_envvec(design_lists, cut_weight_lists, device, rwd, train_num=12, test_num=4)
    policy_kw = make_policy()

    logdir = os.path.join('log_training', datetime.datetime.now().strftime("%Y%m%d") + "_psm")
    checkdir = os.path.join('checkpoints', datetime.datetime.now().strftime("%Y%m%d") + "_psm")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(checkdir):
        os.makedirs(checkdir)

    model = PPO(
        policy = 'MultiInputPolicy',
        env = train_envs,
        learning_rate= cosine_schedule(optim_lr),
        policy_kwargs= policy_kw,
        verbose = 1,
        n_steps= 300,
        batch_size=300,
        n_epochs=5,
        device=device,
        tensorboard_log=logdir,
        ent_coef=0.1,
        clip_range=0.2,
        gamma=0.95,
        vf_coef=0.1,
        use_psm=True,
        psm_coef=0.3,
    )
    model.set_parameters(checkpoint_path, device=device)

    eval_cback = EvalCallback(test_envs, eval_freq=1200,
                n_eval_episodes=20,
                best_model_save_path=checkdir+'/best/')
    check_cback = CheckpointCallback(
        save_freq=300, save_path=checkdir,
        name_prefix="psm_model_point"
    )

    model.learn(total_timesteps=more_timesteps, callback=[eval_cback, check_cback], reset_num_timesteps=False)

    train_envs.close()
    test_envs.close()

import os
import glob
import time
from datetime import datetime
import torch
import numpy as np

def get_config():
    config = {
        "env_name": "Walker2d-v4",
        "has_continuous_action_space": True,
        "max_ep_len": 1000,
        "max_training_timesteps": int(3e6),
        "print_freq": 10000,  # max_ep_len * 10
        "log_freq": 2000,     # max_ep_len * 2
        "save_model_freq": int(1e5),
        "action_std": 0.6,
        "action_std_decay_rate": 0.05,
        "min_action_std": 0.1,
        "action_std_decay_freq": int(100),
        "update_timestep": 4000,  # max_ep_len * 4
        "K_epochs": 80,
        "eps_clip": 0.2,
        "gamma": 0.99,
        "lr_actor": 0.0003,
        "lr_critic": 0.001,
        "random_seed": 0,
        "eposide_save":10
    }
    return config

def print_hyperparameters(config, state_dim, action_dim):
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", config["max_training_timesteps"])
    print("max timesteps per episode : ", config["max_ep_len"])
    print("model saving frequency : " + str(config["save_model_freq"]) + " timesteps")
    print("log frequency : " + str(config["log_freq"]) + " timesteps")
    print("printing average reward over episodes in last : " + str(config["print_freq"]) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if config["has_continuous_action_space"]:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", config["action_std"])
        print("decay rate of std of action distribution : ", config["action_std_decay_rate"])
        print("minimum std of action distribution : ", config["min_action_std"])
        print("decay frequency of std of action distribution : " + str(config["action_std_decay_freq"]) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(config["update_timestep"]) + " timesteps")
    print("PPO K epochs : ", config["K_epochs"])
    print("PPO epsilon clip : ", config["eps_clip"])
    print("discount factor (gamma) : ", config["gamma"])
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", config["lr_actor"])
    print("optimizer learning rate critic : ", config["lr_critic"])
    if config["random_seed"]:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", config["random_seed"])
    print("============================================================================================")

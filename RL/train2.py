import os
import glob
import time
from datetime import datetime
import torch
import numpy as np
import gym
# import roboschool
from models.PPO import PPO
from config.config import get_config,print_hyperparameters

################################### Training ###################################


def train():
    print("============================================================================================")
    # Load configuration
    config = get_config()
    print("training environment name : " + config["env_name"])
    env = gym.make(config["env_name"])
    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if config["has_continuous_action_space"]:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################
    log_dir = "Results/PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + config["env_name"] + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    run_num = len(next(os.walk(log_dir))[2])
    log_f_name = log_dir + '/PPO_' + config["env_name"] + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + config["env_name"] + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0
    directory = "Results/PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + config["env_name"] + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(config["env_name"], config["random_seed"], run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    ############# print all hyperparameters #############
    print_hyperparameters(config, state_dim, action_dim)

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, config["lr_actor"], config["lr_critic"], config["gamma"], config["K_epochs"], config["eps_clip"], config["has_continuous_action_space"], config["action_std"])

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= config["max_training_timesteps"]:

        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1, config["max_ep_len"] + 1):

            # select action with policy
            action = ppo_agent.select_action(state)
            print(action)
            state, reward, done, _, _ = env.step(action)

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % config["update_timestep"] == 0:
                ppo_agent.update()

            # decay action std if needed
            if config["has_continuous_action_space"] and time_step % config["action_std_decay_freq"] == 0:
                ppo_agent.decay_action_std(config["action_std_decay_rate"], config["min_action_std"])

            # log reward
            if time_step % config["log_freq"] == 0:
                log_avg_reward = round(log_running_reward / log_running_episodes, 4)
                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()
                log_running_reward = 0
                log_running_episodes = 0

            # print average reward
            if time_step % config["print_freq"] == 0:
                print_avg_reward = round(print_running_reward / print_running_episodes, 2)
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % config["save_model_freq"] == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
    
    
    
    
    
    
    

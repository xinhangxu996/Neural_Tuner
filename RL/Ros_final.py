import os
import time
from datetime import datetime
import torch
import numpy as np
import rospy
from models.PPO import PPO
from config.config import get_config, print_hyperparameters
from neural_tuner.msg import Observation
from std_msgs.msg import Int32
from std_msgs.msg import Float32MultiArray
import os

class PPO_Trainer:
    def __init__(self):
        rospy.init_node('PPO_Trainer', anonymous=True)
        self.save_directory = os.path.expanduser('~/ICRA_2024/gym_ws/src/neural_tuner/RL/Results/weights')
        # Create the directory if it doesn't exist
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        rospy.sleep(1)
        self.latent_dim = 256  # Set latent dimension as per your model's requirement
        self.action_dim = 6  # Set action dimension as per your ROS environment
        #config
        self.config = get_config()
        self.brain=PPO(self.latent_dim, self.action_dim, self.config["lr_actor"], self.config["lr_critic"], self.config["gamma"], self.config["K_epochs"], self.config["eps_clip"], self.config["has_continuous_action_space"], self.config["action_std"])
        self.brain.load('/home/xxh/ICRA_2024/gym_ws/src/neural_tuner/RL/Results/weights_5x5_v3/episode_840_reward_-250.10.pth')

        self.maximum_game_time=60 #30 second 
        self.hz=1
        self.maximum_step=60
        
        #Game loop related
        self.game_start_time=0
        self.running=False
        self.total_reward=0

        self.state_sub=rospy.Subscriber('/obs_state', Observation, self.state_callback)
        self.reset_pub = rospy.Publisher('/reset_command', Int32, queue_size=10)
        self.run_pub = rospy.Publisher('/env_run', Int32, queue_size=10)
        self.action_pub=rospy.Publisher('/action_array',Float32MultiArray,queue_size=10)
        rospy.sleep(1)
        self.general_msg=Int32()
        self.general_msg.data=1

        self.num_step=0

        self.eposide=840

        self.reset()
        rospy.sleep(0.5)
        self.run()

        rospy.spin()

    def reset(self):

        self.reset_pub.publish(self.general_msg)
        rospy.loginfo("Reset_env")
        self.running=False
        self.num_step=0
        self.total_reward=0

    def run(self):
        self.eposide=self.eposide+1
        self.run_pub.publish(self.general_msg)
        self.running=True
        self.running_time=rospy.Time.now().to_sec()
        rospy.loginfo("Run")
        self.total_reward=0
    
    def state_callback(self,msg):
        if not self.running:
            return
        
        self.num_step=self.num_step+1

        current_state=list(msg.current_people_map)+list(msg.current_obs_map) +list(msg.current_viechle_vel)+[msg.current_viechle_yaw]

        current_reward=msg.reward
        need_reset=msg.need_reset

        if not need_reset and rospy.Time.now().to_sec()-self.running_time>self.maximum_game_time and self.running: 
            need_reset=True
            current_reward=current_reward-1500
        if not need_reset and self.num_step>=self.maximum_step:
            need_reset=True
            current_reward=current_reward-1500

        if need_reset:
            self.brain.buffer.rewards.append(current_reward)
            self.brain.buffer.is_terminals.append(need_reset)
            self.brain.update()
            self.total_reward=self.total_reward+current_reward
            # decay action std if needed
            if self.config["has_continuous_action_space"] and self.eposide % self.config["action_std_decay_freq"] == 0:
                self.brain.decay_action_std(self.config["action_std_decay_rate"], self.config["min_action_std"])
            
            if self.eposide %self.config["eposide_save"]==0:

                # Generate the filename using episode and reward
                filename = f'episode_{self.eposide}_reward_{self.total_reward:.2f}.pth'
                
                # Complete path to save the file
                save_path = os.path.join(self.save_directory, filename)
                self.brain.save(save_path)
            rospy.loginfo(f'Eposide_{self.eposide}_reward_{self.total_reward}_Time:{rospy.Time.now().to_sec()-self.running_time} sec!')
            self.reset()
            rospy.sleep(0.5)
            self.run()

        else:
            
            action = self.brain.select_action(current_state)
            if self.num_step>1:
                
                self.brain.buffer.rewards.append(current_reward)
                
                self.brain.buffer.is_terminals.append(need_reset)

                self.total_reward=self.total_reward+current_reward

            action_msg=Float32MultiArray()
            action_msg.data=action#[1,1,3,6,1]#action
            self.action_pub.publish(action_msg)

            # decay action std if needed
            # if self.config["has_continuous_action_space"] and self.eposide % self.config["action_std_decay_freq"] == 0:
            #     self.brain.decay_action_std(self.config["action_std_decay_rate"], self.config["min_action_std"])

if __name__ == '__main__':
    try:
        # Create an instance of the class and run the monitor method
        PPO_mode  = PPO_Trainer()
    except rospy.ROSInterruptException:
        pass
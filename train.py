# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:27:21 2023

@author: woill
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from datetime import datetime

from DQN import DQNAgent
from game import Game_Env 
from Model_Builder import get_model_builder
from curriculum.alp_gmm import ALPGMM
# from utils import plot_run_avg, plot_mutliple_runnin_avg
import argparse
from utils import load_config, get_param

from gym.spaces import Box

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil

import tensorflow as tf
tf.logging.set_verbosity('ERROR')
from keras.backend.tensorflow_backend import set_session


def train_model(model_name, env, agent, curriculum_manager, tb_filewriter, args_training) : 
    
    # rewards = [] l
    # loss_values, nbs_of_steps, doabilities = [], [], []
    # wall_ratios_spawn, key_elem_ratio_spawn, max_nb_steps_ratios = [], [], []
    train_start_time = time.time()
    
    print(f"Starting Training, model name: {model_name}")
    for e in tqdm(range(args_training.EPISODES+1)):
        
        current_task = curriculum_manager.sample_task()
        wall_ratio = current_task[0]; key_elem_ratio = current_task[1]#; max_nb_steps_ratio = current_task[2]
        state = env.reset(wall_ratio = wall_ratio, key_elem_ratio = key_elem_ratio) #, max_nb_steps_ratio = max_nb_steps_ratio)
        state = np.reshape(state, [1, -1])
        
        done = False
        times_per_step, rewards_ep, q_vals = [], [], []
        while not done: 
            start_step_time = time.time()
            action, max_q_val = agent.act(state, return_q_value = True, use_softmax = args_training.action_select_softmax)
            next_state, reward, done, infos = env.step(action)
            rewards_ep.append(reward)
            if max_q_val is not None : q_vals.append(max_q_val)
            next_state = np.reshape(next_state, [1, -1])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            end_step_time = time.time()
            times_per_step.append(end_step_time - start_step_time)
            # if e > args_training.nb_of_warm_up_ep : agent.replay(args_training.batch_size, update_epsilon = False)
            if done:
                curriculum_manager.update(task=current_task, reward=reward)
                break
        
        summary = tf.Summary()
        if e > args_training.nb_of_warm_up_ep :
            start_replay_time = time.time()
            loss_value = agent.replay(args_training.batch_size)
            end_replay_time = time.time()
            summary.value.add(tag="Model/Loss", simple_value= loss_value)
            summary.value.add(tag="Model/Time Per Replay", simple_value= end_replay_time - start_replay_time)
            if e%args_training.save_weight_every_nb_ep==0:
                agent.save(model_name)
            
        if e % args_training.update_target_network_ep == 0:
            agent.target_train()
        
        
        
        ### Reporting
        summary.value.add(tag="Perfs/RewardEp", simple_value= np.sum(rewards_ep))
        summary.value.add(tag="Perfs/RewardMean", simple_value= np.mean(rewards_ep))
        summary.value.add(tag="Perfs/Nb Steps", simple_value= infos["nb_steps"])
        summary.value.add(tag="Perfs/Doability", simple_value= infos["doable"])
        summary.value.add(tag="Perfs/Epsilon", simple_value= agent.epsilon)
        if q_vals != [] : summary.value.add(tag="Model/Avg Q Value", simple_value= np.mean(q_vals))
        summary.value.add(tag="Model/Time Per Step", simple_value= np.mean(times_per_step))
        summary.value.add(tag="Curricululm/Wall Spawn Ratio", simple_value= wall_ratio)
        summary.value.add(tag="Curricululm/Elems Spawn Ratio", simple_value= key_elem_ratio)
        # summary.value.add(tag="Curricululm/Max Nb Steps Ratio", simple_value= max_nb_steps_ratio)
        tb_filewriter.add_summary(summary, e)
        tb_filewriter.flush()
                
        
        if e%500==0:
            print("episode: {}/{}, score: {:.2}, doable: {}, e: {:.2}"
                  .format(e, args_training.EPISODES, reward, infos["doable"], agent.epsilon))
            env.display()
        
    train_end_time = time.time()
    print("Training Took", (train_end_time - train_start_time)//60, "minutes")
    
    
    return model_name
    


if __name__ == "__main__" :
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7 # fraction of memory
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    
    set_session(tf.Session(config=config))
    
    
    
    parser = argparse.ArgumentParser(description='Run DQN training')
    parser.add_argument('--config-fn', '-f', type=str, default='train_config', help='config file name')
    parser.add_argument('--config-dir', '-d', type=str, default="./Config/", help='config files directory')
    args = parser.parse_args()
    
    
    loaded_config = load_config(args.config_fn, args.config_dir)
    args_training = loaded_config["args_training"]
    
    
    
    model_name = args_training.model_name
    model_name += f"_{datetime.now().strftime('%b_%d_%H_%M_%S')}"
    
    
    ## Init env
    curriculum_manager = ALPGMM(mins = [0.2, 0.2], maxs = [1.0, 1.0], params = {"fit_rate" : 500,
                                                                                "random_task_ratio" : 0.2 if args_training.enable_auto_curriculum else 1.0}) #, "alp_buffer_size" : 1500})
    env = Game_Env(map_size = args_training.map_size, max_nb_steps_ratio = args_training.max_nb_steps_ratio)
    
    ## Init Model and agent
    model_builder = get_model_builder(env, args_training) ## FullyConv  FractalNet
    agent = DQNAgent(state_size = env.state_size, action_size = env.action_size, model_builder = model_builder, args_training = args_training)
    
    
    ## Init Tensorboard reporting
    tb_filewriter = tf.summary.FileWriter(f".\\reporting\\{model_name}")
    ## tensorboard --logdir="C:\Users\woill\Documents\Python Scripts\IMKI\game_gen\reporting"
    
    
    train_model(model_name, env, agent, curriculum_manager, tb_filewriter, args_training)
    

    
    
    
    
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:28:14 2023

@author: woill
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time

from DQN import DQNAgent
from game import Game_Env 
from Model_Builder import get_model_builder
from utils import plot_over_max_nb_steps_ratio, plot_mutliple_over_max_nb_steps_ratio

import argparse
from utils import load_config, get_param

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import importlib
import gif

def test_model(EPISODES, env, agent, wall_ratio = 0.65, key_elem_ratio = 0.75):
    
    rewards = []
    gen_maps, all_rewards = {}, {}
    doabilities = []
    path_lengts_S_to_T, path_lengts_T_to_F = [], []
    
    
    for max_nb_steps_ratio in np.arange(0.1,1.05,0.1) :
        env.set_max_nb_steps_ratio(max_nb_steps_ratio = max_nb_steps_ratio)
        gen_maps[max_nb_steps_ratio] = []
        all_rewards[max_nb_steps_ratio] = []
        for e in range(EPISODES):
            state = env.reset(test = True, wall_ratio = wall_ratio, key_elem_ratio = key_elem_ratio)
            state = np.reshape(state, [1, -1])
            done = False
            
            while not done: 
                action = agent.act(state, use_softmax=False)
                next_state, reward, done, infos = env.step(action)
                state = np.reshape(next_state, [1, -1])
                if done:
                    break
            
            ### Reporting
            rewards.append([max_nb_steps_ratio, reward])
            doabilities.append([max_nb_steps_ratio, infos["doable"]])
            gen_maps[max_nb_steps_ratio].append([state])
            all_rewards[max_nb_steps_ratio].append([reward])
            
            is_doable, path_S_to_T, path_T_to_F = env.is_level_doable()
            if is_doable:
                path_lengts_S_to_T.append([max_nb_steps_ratio, len(path_S_to_T)])
                path_lengts_T_to_F.append([max_nb_steps_ratio, len(path_T_to_F)])
            
        print("Ratio: {:.2} Avg Doability: {:.2}"
              .format(max_nb_steps_ratio, np.mean(np.array(doabilities)[-EPISODES:,1])))
        
    dict_to_plot = {"Doable":doabilities, "Rewards":rewards, "Lenght S to T":path_lengts_S_to_T, "Lenght T to F":path_lengts_T_to_F}
    plot_mutliple_over_max_nb_steps_ratio(dict_to_plot, title = "Perfs for varying number of avaiable steps")
    
    
    return gen_maps, all_rewards


if __name__ == "__main__" :
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7 # fraction of memory
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    
    set_session(tf.Session(config=config))
    
    
    

    
    parser = argparse.ArgumentParser(description='Test DQN Model')
    parser.add_argument('--model-name', '-m', type=str, 
                        default='Map8_CurriculumMoreElemSpawn_SeparatePosAndElemSoftmaxAction_RwrdDistStoF__Mar_20_01_49_34',
                        help='Name of model to test')
    parser.add_argument('--EPISODES', '-E', type=int,  default=50, help='Nb of episodes to run')
    args = parser.parse_args()
    
    
    EPISODES = args.EPISODES
    model_name_saved = args.model_name
    
    ## Map8_SeparatePosAndElemSoftmaxAction_RwrdDistStoFsparserMap__Mar_20_15_57_08 potential to be crazy good
    ## Map8_CurriculumMoreElemSpawn_SeparatePosAndElemSoftmaxAction_RwrdDistStoF__Mar_20_01_49_34 best so far
    ## Map8_SeparatePosAndElemSoftmaxAction_RwrdDistStoF_BigNet__Mar_20_12_37_38
    ## Map8_CurriculumMoreElemSpawn_NoBiasLastLayer__Mar_19_12_30_43 
    ## Map8_CurriculumElemSpawn__Mar_18_19_46_11
    
    
    
    # parser = argparse.ArgumentParser(description='Test DQN Model')
    # parser.add_argument('--config-fn', '-f', type=str, default='config', help='config file name')
    # parser.add_argument('--config-dir', '-d', type=str, default=f"./models/{model_name_saved}/", help='config files directory')
    # args = parser.parse_args()

    
    
    loaded_config = load_config('config', f"./models/{model_name_saved}/")
    args_training = loaded_config["args_training"]
    
    EPISODES = 50
    
    env = Game_Env(map_size = args_training.map_size)
    
    ## Import model builder class
    mod = importlib.import_module(f"models.{model_name_saved}.Model_builder")    
    model_builder = mod.get_model_builder(env, args_training) 
    agent = DQNAgent(state_size = env.state_size, action_size = env.action_size, model_builder = model_builder, args_training = args_training)
    
    ## Load agent
    agent.load(f"models\\{model_name_saved}\\Model_weights.h5")
    agent.set_epsilon(0.00)
    
    
    gen_maps, all_rewards = test_model(EPISODES, env, agent, wall_ratio = 0.65, key_elem_ratio=0.5)


    k_best = 5
    maps = np.array(list(gen_maps.values())).reshape(-1, args_training.map_size**2)
    rwrds = np.array(list(all_rewards.values())).reshape(-1)
    ind_k_vest_rewrd = np.argpartition(rwrds, -k_best)[-k_best:]
    best_rwrds_values = rwrds[ind_k_vest_rewrd]
    print(f"Avg Best Rwrd: {round(np.mean(best_rwrds_values),3)}")
    for index in ind_k_vest_rewrd:
        env.set_current_map(maps[index])
        env.display()
    
    
    
    
    
    
    
    
    """
    k_best = 3
    for max_step_value, rewards in all_rewards.items() :
        gen_state = gen_maps[max_step_value]
        ind_k_vest_rewrd = np.argpartition(np.array(rewards).reshape(-1), -k_best)[-k_best:]
        best_rwrds_values = np.array(rewards).reshape(-1)[ind_k_vest_rewrd]
        print(f"Ratio: {round(max_step_value, 2)} Avg Best Rwrd: {round(np.mean(best_rwrds_values),3)}")
        for index in ind_k_vest_rewrd:
            env.set_current_map(np.array(gen_state)[index][0][0])
            env.display()
    """
            
            



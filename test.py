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
import pandas as pd

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

# def run_test_ep(env, agent):
    


def test_model(EPISODES, env, agent, wall_ratios = [0.65], key_elem_ratios = [0.75], max_nb_steps_ratios = np.arange(0.1,1.05,0.1)):

    gen_maps = []
    df_reporting = pd.DataFrame(columns=["wall_ratio", "key_elem_ratio", "max_nb_steps_ratio", "reward", "is_doable", 
                                         "nb_steps", "len_path_lengts_S_to_T", "len_path_lengts_T_to_F"])
    
    if type(wall_ratios) is float: wall_ratios = [wall_ratios]
    if type(key_elem_ratios) is float: key_elem_ratios = [key_elem_ratios]
    if type(max_nb_steps_ratios) is float: max_nb_steps_ratios = [max_nb_steps_ratios]
    
    for wall_ratio in wall_ratios:
        for key_elem_ratio in key_elem_ratios : 
            for max_nb_steps_ratio in max_nb_steps_ratios:
                for e in range(EPISODES):
                    dict_to_add = {}
                    state = env.reset(test = True, wall_ratio = wall_ratio, key_elem_ratio = key_elem_ratio, max_nb_steps_ratio = max_nb_steps_ratio)
                    dict_to_add["wall_ratio"] = wall_ratio
                    dict_to_add["key_elem_ratio"] = key_elem_ratio
                    dict_to_add["max_nb_steps_ratio"] = max_nb_steps_ratio
                    state = np.reshape(state, [1, -1])
                    done = False
                    
                    while not done: 
                        action = agent.act(state, use_softmax=False)
                        next_state, reward, done, infos = env.step(action)
                        state = np.reshape(next_state, [1, -1])
                        if done:
                            break
                    
                    ### Reporting
                    dict_to_add["reward"] = reward
                    dict_to_add["is_doable"] = infos["doable"]
                    dict_to_add["nb_steps"] = infos["nb_steps"]
        
                    gen_maps.append(state)
                    
                    is_doable, path_S_to_T, path_T_to_F = env.is_level_doable()
                    if is_doable:
                        dict_to_add["len_path_lengts_S_to_T"] = len(path_S_to_T)
                        dict_to_add["len_path_lengts_T_to_F"] = len(path_T_to_F)
                        
                    df_reporting = df_reporting.append(dict_to_add, ignore_index=True)    
                
                print("Wall: {:.2}, KeyElem: {:.2}, MaxSteps: {:.2} Avg Doability: {:.2}"
                      .format(wall_ratio, key_elem_ratio, max_nb_steps_ratio, np.mean(df_reporting["is_doable"].values[-EPISODES:])))
        

    
    
    return gen_maps, df_reporting


if __name__ == "__main__" :
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7 # fraction of memory
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    
    set_session(tf.Session(config=config))
    
    
    

    
    parser = argparse.ArgumentParser(description='Test DQN Model')
    parser.add_argument('--model-name', '-m', type=str, 
                        default='Map8_Unet_NewRwrd__Mar_22_19_57_27',
                        help='Name of model to test')
    parser.add_argument('--checkpoint-nb', '-cn', type=int,  default=None, help='which checkpoint to load')
    parser.add_argument('--EPISODES', '-E', type=int,  default=50, help='Nb of episodes to run')
    args = parser.parse_args()
    
    
    EPISODES = args.EPISODES
    model_name_saved = args.model_name
    
    ## Map8_RwrdDistStoFsparserMap_EpsMin01_DuelingDQNAvgNoMask_ClipGradNorm1__Mar_21_13_22_15
    ## Map8_SeparatePosAndElemSoftmaxAction_RwrdDistStoFsparserMap__Mar_20_15_57_08 potential to be crazy good
    ## Map8_CurriculumMoreElemSpawn_SeparatePosAndElemSoftmaxAction_RwrdDistStoF__Mar_20_01_49_34 best so far
    ## Map8_SeparatePosAndElemSoftmaxAction_RwrdDistStoF_BigNet__Mar_20_12_37_38
    ## Map8_CurriculumMoreElemSpawn_NoBiasLastLayer__Mar_19_12_30_43 
    ## Map8_CurriculumElemSpawn__Mar_18_19_46_11
    
        
    loaded_config = load_config('config', f"./models/{model_name_saved}/")
    args_training = loaded_config["args_training"]
    
    EPISODES = 50
    
    env = Game_Env(map_size = args_training.map_size)
    
    ## Import model builder class
    mod = importlib.import_module(f"models.{model_name_saved}.Model_builder")    
    model_builder = mod.get_model_builder(env, args_training) 
    agent = DQNAgent(state_size = env.state_size, action_size = env.action_size, model_builder = model_builder, args_training = args_training)
    
    ## Load agent
    agent.load(model_name_saved, args.checkpoint_nb)
    agent.set_epsilon(0.05)
    
    wall_ratios = np.arange(0.1,1.05,0.2)
    key_elem_ratios = [0.5]
    max_nb_steps_ratios = np.arange(0.1,1.05,0.2)
    
    gen_maps, df_reporting = test_model(EPISODES, env, agent, wall_ratios = wall_ratios, key_elem_ratios = key_elem_ratios, max_nb_steps_ratios = max_nb_steps_ratios)

    for col_name in ["max_nb_steps_ratio", "wall_ratio"] : 
        dict_to_plot = {"Doable": np.array(df_reporting[[col_name, "is_doable"]].values, dtype = np.float32),
                        "Rewards": np.array(df_reporting[[col_name, "reward"]].values, dtype = np.float32),
                        "Lenght S to T": np.array(df_reporting[[col_name, "len_path_lengts_S_to_T"]].values, dtype = np.float32),
                        "Lenght T to F": np.array(df_reporting[[col_name, "len_path_lengts_T_to_F"]].values, dtype = np.float32)
                        }
        plot_mutliple_over_max_nb_steps_ratio(dict_to_plot, title = f"Perfs for varying number of {col_name}")


    k_best = 5
    maps = np.array(gen_maps).reshape(-1, args_training.map_size**2)
    best_maps= df_reporting.nlargest(n = k_best, columns = "reward")
    print(f"Avg Best Rwrd: {round(np.mean(best_maps.reward),3)}")
    for index in best_maps.index:
        env.set_current_map(maps[index])
        env.display()
    
    
    if len(df_reporting.max_nb_steps_ratio.unique()) > 1 and len(df_reporting.wall_ratio.unique()) > 1:
        df_reporting.is_doable = df_reporting.is_doable.astype(float)
        for col_to_agg in ["reward", "is_doable"]:
            table = pd.pivot_table(np.round(df_reporting, 2), values=col_to_agg, index=['wall_ratio'],
                        columns=['max_nb_steps_ratio'], aggfunc=np.mean)
            sns.heatmap(table)
            plt.title(f"Avg {col_to_agg}")
            plt.show()
        
        
        
        



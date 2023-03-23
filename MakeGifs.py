# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 23:31:54 2023

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

@gif.frame
def display_for_gif(env, include_paths, show, title):
    env.display(include_paths = include_paths, show = show, title = title)

def make_gifs(EPISODES, env, agent, wall_ratio = 0.65, key_elem_ratio = 0.75, model_name_saved = "", gif_name = None):
    ## What is commented out is to make the best gifs, for reporting.
    
    frames = []
    for e in range(EPISODES):
        state = env.reset(test = True, wall_ratio = wall_ratio, key_elem_ratio = key_elem_ratio)
        for _ in range(2) : frames.append(display_for_gif(env, include_paths = True, show = False, title = f"Ep {e} ||"))

    # frames_cache = []
    # ep = 1
    # while ep <= 10 :
    #     frame_ep = []
    #     state = env.reset(test = True, wall_ratio = np.random.choice([0.25, 0.5, 0.75]), key_elem_ratio = np.random.choice([0.2, 0.5, 1.0]), max_nb_steps_ratio = np.random.choice([0.2, 0.5, 0.75]))
    #     for _ in range(2) : frame_ep.append(display_for_gif(env, include_paths = True, show = False, title = f"Ep {ep} ||"))
        
        
        state = np.reshape(state, [1, -1])
        done = False
        
        while not done: 
            action = agent.act(state, use_softmax=False)
            next_state, reward, done, infos = env.step(action)
            state = np.reshape(next_state, [1, -1])
            frames.append(display_for_gif(env, include_paths = False, show = False, title = f"Ep {e} ||"))
            # frame_ep.append(display_for_gif(env, include_paths = False, show = False, title = f"Ep {ep} ||"))
            if done:
                for _ in range(3) : frames.append(display_for_gif(env, include_paths = True, show = False, title = f"Ep {e} ||"))
                # for _ in range(3) : frame_ep.append(display_for_gif(env, include_paths = True, show = False, title = f"Ep {ep} ||"))
                break
        
        # if reward > 0.65 :
        #     print(ep, reward)
        #     frames_cache += frame_ep
        #     ep += 1
    
    if gif_name is None : 
        gif_name = "examples"
    gif.save(frames, f"reporting\\{model_name_saved}\\{gif_name}.gif", duration=250)
    # gif.save(frames_cache, f"reporting\\{model_name_saved}\\{gif_name}.gif", duration=250)


if __name__ == "__main__" :
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2 # fraction of memory
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    
    set_session(tf.Session(config=config))
    
    
    parser = argparse.ArgumentParser(description='Make Gifs After Training')
    parser.add_argument('--model-name', '-m', type=str, 
                        default='Map8_SmallNet_Unet_NewRwrd__Mar_23_14_22_28',
                        help='Name of model to test')
    parser.add_argument('--checkpoint-nb', '-cn', type=int,  default=None, help='which checkpoint to load')
    parser.add_argument('--EPISODES', '-E', type=int,  default=10, help='Nb of episodes to run')
    args = parser.parse_args()
    
    
    EPISODES = args.EPISODES
    model_name_saved = args.model_name
    
    
    loaded_config = load_config('config', f"./models/{model_name_saved}/")
    args_training = loaded_config["args_training"]
    
    
    env = Game_Env(map_size = args_training.map_size)
    
    ## Import model builder class
    mod = importlib.import_module(f"models.{model_name_saved}.Model_builder")    
    model_builder = mod.get_model_builder(env, args_training) 
    agent = DQNAgent(state_size = env.state_size, action_size = env.action_size, model_builder = model_builder, args_training = args_training)
    
    ## Load agent
    agent.load(model_name_saved, args.checkpoint_nb)
    
    
    wall_ratio = 0.5
    key_elem_ratio = 0.5
    max_nb_steps_ratio = 0.5
    agent.set_epsilon(0.1)

    
    env.set_max_nb_steps_ratio(max_nb_steps_ratio = max_nb_steps_ratio)
    make_gifs(EPISODES, env, agent, wall_ratio = wall_ratio, key_elem_ratio=key_elem_ratio, model_name_saved = model_name_saved, gif_name = f"Wr{wall_ratio}_Er{key_elem_ratio}_MSr{max_nb_steps_ratio}_Eps{agent.epsilon}")





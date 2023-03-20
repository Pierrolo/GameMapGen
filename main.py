# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:55:42 2023

@author: woill
"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from DQN import DQNAgent
from game import Game_Env 
from Model_Builder import get_model_builder
from utils import plot_run_avg, plot_mutliple_runnin_avg

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from train import train_model
from test import test_model



if __name__ == "__main__" :


    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7 # fraction of memory
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    
    set_session(tf.Session(config=config))


    ## TRAIN
    model_name = "map8_ratiofinish025"
    
    map_size = 8
    learning_rate = 0.0001
    
    batch_size = 256
    EPISODES = 100000
    
    model_name_saved = train_model(model_name, map_size, learning_rate, batch_size, EPISODES)
    
    
    
    
    ## TEST
    
    model_name_saved = "map8_ratiofinish025_100K"
    map_size = 8
    EPISODES = 100
    gen_maps = test_model(model_name_saved, map_size, EPISODES)
    
    

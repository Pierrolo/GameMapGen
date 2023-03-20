# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 13:50:57 2023

@author: woill
"""
import os
import yaml
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_run_avg(y, window = 100, title = "", ax = None):
    average_y = []
    for ind in range(len(y) - window + 1):
        average_y.append(np.mean(y[ind:ind+window]))
    
    if ax is None :
        plt.plot(average_y)
        plt.title(title)
        plt.show()
    else : 
        ax.plot(average_y)
        ax.set_title(title)



def plot_mutliple_runnin_avg(dict_to_plot, window = 100, title = "Training Recap"):
    fig, axs = plt.subplots(1, len(dict_to_plot), figsize = (len(dict_to_plot)*6,5))
    axs.reshape(-1)
    ax_nb = 0
    
    for title_, vals in dict_to_plot.items():
        plot_run_avg(vals, window = window, title = title_, ax = axs[ax_nb])
        ax_nb += 1
        
    fig.suptitle(title)
    plt.show()




def plot_over_max_nb_steps_ratio(list_to_plot, title= "", ax = None):
    array_to_plot = np.array(list_to_plot)
    sns.lineplot(x = array_to_plot[:,0], y = array_to_plot[:,1], ax=ax)
    if ax is None : 
        plt.title(title)
        plt.show()



def plot_mutliple_over_max_nb_steps_ratio(dict_to_plot, title = "Perfs for varying number of avaiable steps"):
    
    
    fig, axs = plt.subplots(1, len(dict_to_plot), figsize = (len(dict_to_plot)*6,5))
    axs.reshape(-1)
    ax_nb = 0
    
    for title_, vals in dict_to_plot.items():
        plot_over_max_nb_steps_ratio(vals, ax = axs[ax_nb])
        axs[ax_nb].set_title(title_)
        ax_nb += 1
        
    fig.suptitle(title)
    plt.show()
    
    


class AttrDict(dict):

  def __init__(self, *args, **kwargs):
    unlocked = kwargs.pop('_unlocked', not (args or kwargs))
    defaults = kwargs.pop('_defaults', {})
    touched = kwargs.pop('_touched', set())
    super(AttrDict, self).__setattr__('_unlocked', True)
    super(AttrDict, self).__setattr__('_touched', set())
    super(AttrDict, self).__setattr__('_defaults', {})
    super(AttrDict, self).__init__(*args, **kwargs)
    super(AttrDict, self).__setattr__('_unlocked', unlocked)
    super(AttrDict, self).__setattr__('_defaults', defaults)
    super(AttrDict, self).__setattr__('_touched', touched)

  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)



def load_json(data_dir, data_fn):
    data_fp = os.path.join(data_dir, data_fn)
    assert os.path.exists(data_fp), "File does not exist : "+data_fp
    with open(data_fp, 'r') as f:
        loaded_data = json.load(f)
    return loaded_data


def load_config(fn, config_dir):
    fp = "{}{}.config".format(config_dir, fn)
    print("\nLoading config file from {}".format(fp))
    with open(fp, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)
    return args


def get_param(params_dict, key):
    try:
        val = params_dict[key]
    except (TypeError, KeyError):
        val = None
    return val

















        
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 23:33:33 2023

@author: woill
"""
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from prio_buffer import PrioritizedReplayBuffer


import os
import shutil


def random_argmax(a):
    max_values=np.flatnonzero(a == np.nanmax(a))
    if max_values.shape[0]==0:
        return 0
    return np.random.choice(max_values)
def softmax(x):
    e_x = np.exp(x - np.nanmax(x))
    return e_x / np.nansum(e_x)


class DQNAgent:
    def __init__(self, state_size, action_size, model_builder = None, args_training = None) :
        
        # capacity=100000, alpha = 0.6, gamma = 0.95, 
        #          epsilon = 1.0, epsilon_min = 0.05, epsilon_decay = 0.9995, learning_rate = 0.001):
        
        self.state_size = state_size
        self.action_size = action_size
        self.args_training = args_training
        
        
        self.memory = PrioritizedReplayBuffer(capacity=self.args_training.capacity, alpha=self.args_training.alpha)
        self.gamma = self.args_training.gamma   # discount rate
        self.epsilon = self.args_training.epsilon  # exploration rate
        self.epsilon_min = self.args_training.epsilon_min
        self.epsilon_decay = self.args_training.epsilon_decay
        self.learning_rate = self.args_training.learning_rate
        if model_builder is not None : 
            self._build_model = model_builder
        self.model = self._build_model()
        self.model.summary()
        self.target_model = self._build_model()

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))


    def get_softmax_ac(self, act_values, state):
        if state is None :  ## No masking
            act_values_softmax=softmax(act_values_softmax)
            a = np.random.choice(act_values_softmax[0],p=act_values_softmax[0])
            action = np.random.choice(np.where(act_values_softmax == a)[1])
        else : 
            curr_state = np.array(state[0], dtype = np.int16)
            action_mask = np.zeros((curr_state.shape[0],5))
            action_mask[np.arange(curr_state.size), curr_state] = 1
            action_mask = action_mask.reshape(1,-1)
            act_values_softmax = act_values.copy()
            act_values_softmax[action_mask==1] = np.nan
            act_values_softmax=softmax(act_values_softmax)
            act_values_softmax[np.isnan(act_values_softmax)]=0
            a = np.random.choice(act_values_softmax[0],p=act_values_softmax[0])
            action = np.random.choice(np.where(act_values_softmax == a)[1])        
        return action


    def get_random_action(self, state = None):
        if state is None : ## No masking
            action = random.randrange(self.action_size)
        else :
            curr_state = np.array(state[0], dtype = np.int16)
            action_mask = np.zeros((curr_state.shape[0],5))
            action_mask[np.arange(curr_state.size), curr_state] = 1
            action=  np.random.choice(np.where(1-action_mask.reshape(-1))[0])
        return action

    def act(self, state, use_softmax = False, return_q_value = False):
        if np.random.rand() <= self.epsilon:
            if use_softmax:
                act_values = self.model.predict(state)  
                action = self.get_softmax_ac(act_values, state = state if self.args_training.mask_useless_action else None)
            else : 
                action = self.get_random_action(state = state if self.args_training.mask_useless_action else None)
            if return_q_value : 
                q_val = None
        else :
            act_values = self.model.predict(state)  
            action = np.argmax(act_values[0]) 
            if return_q_value : 
                q_val = np.max(act_values)
        if return_q_value :  ## for reporting purposes
            return action, q_val
        else : 
            return action


    def replay(self, batch_size, update_epsilon = True):
        experiences, indices, weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = experiences
        targets = self.model.predict(states)
        Q_next_states = self.target_model.predict(next_states)
        targets[np.arange(batch_size), actions] = rewards + self.gamma * np.max(Q_next_states, axis=1) * (1 - dones)
        errors = np.sum(np.abs(targets - self.model.predict(states)), -1) ## sum to remove all useless zeroes
        self.memory.update_priorities(indices, errors + self.memory.epsilon)
        # self.model.fit(states, targets, sample_weight=weights, epochs=1, verbose=0)
        loss_value = self.model.train_on_batch(states, targets, sample_weight=weights)
        self.memory.update_beta()

        if update_epsilon and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss_value


    def target_train(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)


    def load(self, name, e = None):
        if e is None:            
            if len([name for name in os.listdir(f"models\\{name}\\") if name[-2:] == "h5"]) == 1 :
                self.model.load_weights(f"models\\{name}\\Model_weights.h5")
                self.target_model.load_weights(f"models\\{name}\\Model_weights.h5")
            else : 
                e = max([int(name.split(".")[0].split("_")[-1][:-1]) for name in os.listdir(f"models\\{name}\\") if name[-2:] == "h5"])
                self.model.load_weights(f"models\\{name}\\Model_weights_{e}K.h5")
                self.target_model.load_weights(f"models\\{name}\\Model_weights_{e}K.h5")
        else : 
            self.model.load_weights(f"models\\{name}\\Model_weights_{e}K.h5")
            self.target_model.load_weights(f"models\\{name}\\Model_weights_{e}K.h5")


    def save(self, name, e = None):
        if not os.path.exists(f"models\\{name}\\"):
            ## Copy the py builder file
            os.makedirs(f"models\\{name}\\",  exist_ok = True)
            shutil.copyfile(".\\Model_builder.py", f"models\\{name}\\Model_builder.py")
            shutil.copyfile(".\\Config\\train_config.config", f"models\\{name}\\config.config")
        if e is None:
            self.model.save_weights(f"models\\{name}\\Model_weights.h5")
        else :
            self.model.save_weights(f"models\\{name}\\Model_weights_{e//1000}K.h5")




if __name__ == "__main__":
    self = DQNAgent(state_size=4, action_size=4)

    import gym
    
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    EPISODES = 1000
    batch_size = 32
    rewards = []
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        time = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            time += 1
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        rewards.append(time)
        
        if e % 10 == 0:
            agent.target_train()
    
    env.close()
    
    # Save the weights
    agent.save("dqn.h5")



    window = 15
    average_y = []
    for ind in range(len(rewards) - window + 1):
        average_y.append(np.mean(rewards[ind:ind+window]))
        
    plt.plot(average_y)



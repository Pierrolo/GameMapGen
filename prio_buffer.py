# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 23:41:24 2023

@author: woill
"""

import numpy as np
import random
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_annealing=0.9999, epsilon=1e-8):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))
    
    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[i] = p

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum() + self.epsilon

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max() + self.epsilon

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for i in indices:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(np.array(state[0], copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state[0], copy=False))
            dones.append(done)

        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)), indices, weights

    def update_beta(self):
        self.beta = min(1, self.beta / self.beta_annealing)

    def clear(self):
        self.buffer.clear()
        self.priorities.clear()

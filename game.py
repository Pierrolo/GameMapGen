# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 23:12:00 2023

@author: woill
"""
import numpy as np

# import pygame
# import random
import matplotlib.pyplot as plt


import game_param
import A_star
from A_star import a_star



EMPTY = 0
WALL = 1
START = 2
FINISH = 3
TREASURE = 4




class Map(object):
    ## This the the map object, only focused around keeping all the info of the map and with the ability to apply tile change
    def __init__(self, map_size):
        self.map_size = map_size
        
        self.generate_random_map_train()


    def set_map_size(self, map_size):
        self.map_size = map_size
        
    def set_current_map(self, state):
        self.map_array = state


    def spawn_randomly_element(self, elem = EMPTY):
        rdm_pos = tuple(np.random.randint(0, self.map_size, 2))
        self.change_tile(rdm_pos, elem)
        

    def generate_random_map_test(self, wall_ratio = 0.65, key_elem_ratio = 0.75):
        # self.map_array = np.zeros((self.map_size,self.map_size), dtype = np.int16) * EMPTY
        p = np.array([(1-wall_ratio) * ((self.map_size**2 - 3*key_elem_ratio)/(self.map_size**2)),
                      wall_ratio * ((self.map_size**2 - 3*key_elem_ratio)/(self.map_size**2)),
                      key_elem_ratio/(self.map_size**2),
                      key_elem_ratio/(self.map_size**2),
                      key_elem_ratio/(self.map_size**2)])
        p = p/sum(p)
        self.map_array = np.random.choice([EMPTY, WALL, START, FINISH, TREASURE], p = p, size = (self.map_size,self.map_size))
                

    def generate_random_map_train(self, wall_ratio = 0.33, key_elem_ratio = 1.0):
        p = np.array([(1-wall_ratio) * ((self.map_size**2 - 3*key_elem_ratio)/(self.map_size**2)),
                      wall_ratio * ((self.map_size**2 - 3*key_elem_ratio)/(self.map_size**2)),
                      key_elem_ratio/(self.map_size**2),
                      key_elem_ratio/(self.map_size**2),
                      key_elem_ratio/(self.map_size**2)])
        p = p/sum(p)
        self.map_array = np.random.choice([EMPTY, WALL, START, FINISH, TREASURE], p = p, size = (self.map_size,self.map_size))


    def change_tile(self, tile_pos, new_elem):
        self.map_array[tuple(tile_pos)] = new_elem
        
    
    def get_unique_counts_of_all_elems(self):
        return dict(np.array(np.unique(self.map_array, return_counts=True), dtype = np.int8).T)
    
    def check_all_key_elements_present(self):
        ## Check if there is 1 entry, 1 exit, and 1 treasure
        unique_tiles = self.get_unique_counts_of_all_elems()
        if not all(item in list(unique_tiles.keys()) for item in [START, FINISH, TREASURE]):
            return False
        else :
            if not unique_tiles[START] == unique_tiles[FINISH] == unique_tiles[TREASURE] == 1:
                return False
        
        return True
    
    
    def get_grid_array_for_path_finding(self, from_ = START, to_ = TREASURE, avoid = []):
        
        grid = self.map_array.copy()
        
        grid[np.where(self.map_array== EMPTY) ] = A_star.EMPTY
        grid[np.where(self.map_array == from_)] = A_star.EMPTY
        grid[np.where(self.map_array == to_)] = A_star.EMPTY
        for to_avoid in avoid : ## avoid = [WALL, FINISH]
            grid[np.where(self.map_array == to_avoid)] = A_star.OBSTACLE

        return grid
        
    
    def get_pos_of_elem(self, elem_id = WALL):
        return np.array(np.where(self.map_array== elem_id)).T
    
    
    
    def compute_display_array(self):
        display_array = np.ones((self.map_size, self.map_size,3))* game_param.WHITE
        
        display_array[np.where(self.map_array == WALL)] = game_param.BLACK
        display_array[np.where(self.map_array == START)] = game_param.BLUE
        display_array[np.where(self.map_array == FINISH)] = game_param.GREEN
        display_array[np.where(self.map_array == TREASURE)] = game_param.YELLOW
        
        return np.array(display_array, dtype = np.int16)
    
    
    def plot_path(self, path, col = "r", delta = 0):
        for step_nb in range(len(path)-1):
            plt.arrow(x=path[step_nb][1]+delta, y=path[step_nb][0]+delta,
                      dx=path[step_nb+1][1]-path[step_nb][1], dy=path[step_nb+1][0]-path[step_nb][0],
                      head_width=0.25, head_length=0.2, width = 0.05, fc='w', ec=col, length_includes_head = True)

        
    def display(self, quality = 0, paths = [None, None], show = True, title =""):
        ## display the map into a plot
        plt.figure(figsize = (5,5))
        display_array  = self.compute_display_array()
        plt.imshow(display_array)
        
        path_S_to_T, path_T_to_F = paths[0], paths[1]
        


        if path_S_to_T is not None : self.plot_path(path_S_to_T, col = "y", delta = 0.1)
        if path_T_to_F is not None : self.plot_path(path_T_to_F, col = "g", delta = -0.1)

        plt.yticks(np.arange(self.map_size), np.arange(self.map_size))
        plt.xticks(np.arange(self.map_size), np.arange(self.map_size))
        for letter, elem_id  in {"S" : START, "F" : FINISH, "T": TREASURE, "W": WALL}.items() : 
            elem_positions = np.where(self.map_array == elem_id)
            for elem_nb in range(elem_positions[0].shape[0]):
                elem_positions[0][elem_nb]
                elem_positions[1][elem_nb]
                plt.text(elem_positions[1][elem_nb], elem_positions[0][elem_nb], letter,
                         color = "white", fontsize = 18, fontweight="bold", horizontalalignment = "center")
        

        plt.title(f"{title} Current Map, quality: {round(quality, 3)}")
        
        if show :
            plt.show()






class Game_Env(object):
    ## This is the game env, with which the DQN will interact
    ## action is of type "wide", following "PCGRL: Procedural Content Generation via Reinforcement Learning" (2020)
    
    def __init__(self, map_size = 4, max_nb_steps_ratio = 0.25):
        self.all_elements = [EMPTY, WALL, START, FINISH, TREASURE]
        self.map_size = map_size
        self.max_nb_steps_ratio = max_nb_steps_ratio
        
        self.init_new_map()
        
        self.state_size = self.map_size*self.map_size
        self.action_size = self.map_size*self.map_size * len(self.all_elements)
        
        self.reset()
    
    def set_map_size(self, map_size):
        self.map_size = map_size
        self.current_map.set_map_size(map_size)


    def set_max_nb_steps_ratio(self, max_nb_steps_ratio):
        self.max_nb_steps_ratio = max_nb_steps_ratio

    
    def set_current_map(self, state):
        self.current_map.set_current_map(state.reshape(self.map_size, self.map_size))


    def init_new_map(self):
        self.current_map = Map(map_size = self.map_size)
    
            
    def decode_action_wide(self, action):
        ## action is expressed as the id of the flatten array, need to translate it into pos, elem_id
        tile_pos_x = action//(self.map_size*len(self.all_elements))
        tile_pos_y = (action-tile_pos_x*self.map_size*len(self.all_elements))//len(self.all_elements)
        tile_pos = (tile_pos_x, tile_pos_y)
        elem_id = action%len(self.all_elements)
        
        return tile_pos, elem_id
    
    def encode_action_wide(self, tile_pos, elem_id):
        ## it's the inverse of decode_action_wide
        action = tile_pos[0]*self.map_size*len(self.all_elements)+tile_pos[1]*len(self.all_elements)+elem_id
        return action
    
    def apply_action(self, action):
        ## Apply action on the current_map object
        tile_pos, elem_id = self.decode_action_wide(action)
        self.current_map.change_tile(tile_pos = tile_pos, new_elem = elem_id)
    
    
    def reset(self, wall_ratio = 0.33, key_elem_ratio = 0.5, max_nb_steps_ratio = None, test = False):
        if max_nb_steps_ratio is not None : 
            self.set_max_nb_steps_ratio(max_nb_steps_ratio)
        if test :
            self.current_map.generate_random_map_test(wall_ratio = wall_ratio, key_elem_ratio = key_elem_ratio)
        else :
            self.current_map.generate_random_map_train(wall_ratio = wall_ratio, key_elem_ratio = key_elem_ratio)
        
        self.nb_steps = 0
        self.current_quality = self.get_quality()
        
        next_state = self.get_state()
        return next_state
    
    
    def step(self, action):
        ## action is in range(action_size), it encodes which tiles to be changed, and into what
        self.apply_action(action)
        
        reward = self.get_reward()
        next_state = self.get_state()
        done = self.get_done()
        
        ## Update what needs to be updated each steps
        self.current_quality = self.get_quality()
        infos = self.get_infos(action = action)
        
        self.nb_steps += 1
        
        return next_state, reward, done, infos
        
    
    def get_state(self):
        ## The state is the current map array, flatten
        current_map_array = self.current_map.map_array
        current_map_array_flatten = np.array(current_map_array.reshape(-1), dtype = np.float32)
        state = current_map_array_flatten
        
        
        return state
    
    
    
    
    def get_reward(self):
        reward  = 0.
        # reward = self.get_quality() - self.current_quality
        # reward -= 0.1
        if self.get_done():
            reward += self.get_quality()
        return reward


    def get_infos(self, action = None):
        ## for reporting and such
        infos = {}
        infos["doable"] = self.is_level_doable()[0]
        infos["nb_steps"] = self.nb_steps
        infos["quality"] = self.get_quality()
        
        if action is not None : 
            tile_pos, elem_id = self.decode_action_wide(action)
            infos["tile_played_id"] = elem_id
            
            
        return infos



    def is_level_doable(self):
        ## Is the level doable, if so return the paths as well
        if self.current_map.check_all_key_elements_present() :
            path_S_to_T = self.get_path(from_ = START, to_ = TREASURE, avoid = [WALL, FINISH])
            path_T_to_F = self.get_path(from_ = TREASURE, to_ = FINISH, avoid = [WALL])
            if path_S_to_T is not None and path_T_to_F is not None : 
                return True, path_S_to_T, path_T_to_F            
            
        return False, None, None



    def get_quality(self):
        quality = 0
        
        unique_counts_of_elems = self.current_map.get_unique_counts_of_all_elems()
        one_each_key_element = self.current_map.check_all_key_elements_present()
        
        
        ## Presence of each key element, only once
        for key_elem in [START, FINISH, TREASURE] : 
            if key_elem in unique_counts_of_elems.keys() :
                if unique_counts_of_elems[key_elem] == 1:
                    quality += 0.33
                else : 
                    quality -= 2*unique_counts_of_elems[key_elem] / (self.map_size**2)
        
        
        if not one_each_key_element :
            ## there isn't one and only one of each key element
            quality -= 1.5
        else : 
            path_S_to_T = self.get_path(from_ = START, to_ = TREASURE, avoid = [WALL, FINISH])
            path_T_to_F = self.get_path(from_ = TREASURE, to_ = FINISH, avoid = [WALL])        
            
            if path_S_to_T is None or path_T_to_F is None :
                ## the level is not doable
                quality -= 1.0
            else : 
                # quality += 5*len(np.unique(path_T_to_F + path_S_to_T, axis = 0)) / (self.map_size**2)
                quality += 5*len(path_T_to_F + path_S_to_T) / (self.map_size**2)
                
                if WALL in unique_counts_of_elems.keys():         
                    ## to increase the nb of walls
                    quality += 1.5 * unique_counts_of_elems[WALL]/(self.map_size**2)
                    
        
        
        
        """
        ## If key elements are here, then is there a path between them ?
        if START in unique_counts_of_elems.keys() and unique_counts_of_elems[START] == 1 and TREASURE in unique_counts_of_elems.keys() and unique_counts_of_elems[TREASURE] == 1:
            path_S_to_T = self.get_path(from_ = START, to_ = TREASURE, avoid = [WALL, FINISH])
            if path_S_to_T is None :
                quality -= 1.0
            else : 
                quality += 5*len(path_S_to_T) / (self.map_size * self.map_size)
        else : 
            path_S_to_T = None
            
        if TREASURE in unique_counts_of_elems.keys() and unique_counts_of_elems[TREASURE] == 1 and FINISH in unique_counts_of_elems.keys() and unique_counts_of_elems[FINISH] == 1 :
            path_T_to_F = self.get_path(from_ = TREASURE, to_ = FINISH, avoid = [WALL])        
            if path_T_to_F is None :
                quality -= 1.0
            else : 
                quality += 5*len(path_T_to_F) / (self.map_size * self.map_size)
        else : 
            path_T_to_F = None
            
            
            
        ## Is the level overall doable ?
        if path_S_to_T is not None and path_T_to_F is not None : 
            quality += 2.5
            
            quality += 5*len(np.unique(path_T_to_F + path_S_to_T, axis = 0))/len(path_T_to_F + path_S_to_T)                                       ## compute how much S->T is different from T->F
            if EMPTY in unique_counts_of_elems.keys() and unique_counts_of_elems[EMPTY] != 0:
                quality -= 5*(unique_counts_of_elems[EMPTY] - len(np.unique(path_T_to_F + path_S_to_T, axis = 0))+3)/unique_counts_of_elems[EMPTY]    ## Compute how much useless empty spaces there are
            
            if START in unique_counts_of_elems.keys() and unique_counts_of_elems[START] == 1 and FINISH in unique_counts_of_elems.keys() and unique_counts_of_elems[FINISH] == 1 :
                path_S_to_F = self.get_path(from_ = START, to_ = FINISH, avoid = [WALL])        
                quality += len(path_S_to_F) / (self.map_size * self.map_size)
            
        else : 
            quality -= 2.5
        
        
        # Improve diversity
        if WALL in unique_counts_of_elems.keys():            
            quality += 3 * unique_counts_of_elems[WALL]/(self.map_size * self.map_size)
        """
                
        quality  = quality/5
        
        
        return quality




    def get_done(self):
        ## is the episode done or not
        if self.nb_steps >= self.map_size*self.map_size * self.max_nb_steps_ratio:
            return True
        
        doable, _, _ = self.is_level_doable()
        if doable:
            return True
        
        return False


    def get_path(self, from_ = START, to_ = TREASURE, avoid = [WALL]):
        ## Transform map_array into a pathfinding grid
        grid = self.current_map.get_grid_array_for_path_finding(from_ = from_, to_ = to_, avoid = avoid)
        
        from_pos = tuple(self.current_map.get_pos_of_elem(elem_id = from_)[0])
        to_pos = tuple(self.current_map.get_pos_of_elem(elem_id = to_)[0])
        path = a_star(grid = grid, start = from_pos, goal = to_pos)
        
        return path


    def display(self, include_paths = True, show = True, title = ""):
        ## plot the current map, with the paths is doable
        if include_paths :
            doable, path_S_to_T, path_T_to_F = self.is_level_doable()
        else : 
            doable, path_S_to_T, path_T_to_F = None, None, None
        self.current_map.display(quality = self.get_quality(), paths = [path_S_to_T, path_T_to_F], show = show, title = title)


if __name__ == "__main__":
    
    map_size = 8
    
    
    self = Game_Env(map_size = map_size)
    
    
    self.reset(wall_ratio = 0.5, key_elem_ratio=0.5)
    self.display()
    self.current_map.check_all_key_elements_present()

    

    self.current_map.change_tile((4,4), EMPTY)    


    self.current_map.map_array

    
    self.display()
    
    tile_pos = (2,1) ; elem_id = 1
    action = self.encode_action_wide(tile_pos, elem_id)
    self.apply_action(action = action)
    
    self.current_map.display()


    











    while not self.current_map.check_all_key_elements_present():
        self.reset()







    
    
    aa = np.zeros((8,8,5))    
    for x in range(self.map_size):
        for y in range(self.map_size):
            for z in range(len(self.all_elements)):
                
                action_enc = self.encode_action_wide((x,y), z)
                tile_pos ,z_ = self.decode_action_wide(action_enc)
                
                
                assert((x,y,z) == (tile_pos[0], tile_pos[1], z_))
    
    
                aa = np.zeros((8,8,5))
                
                action = (x,y,z)
                aa[action] = 1.0
                
                bb = aa.reshape(-1)
                ac = np.where(bb)[0][0]
                
                
                if ac != action[0]*8*5+action[1]*5+action[2]:
                    print("error")
                
                
                ac_trasnlate = (ac//(8*5), (ac-(8*5*(ac//(8*5))))//5, ac%5)
                if aa[ac_trasnlate] != 1.0:
                    print("error")
                    
    
    
    
    
    
    
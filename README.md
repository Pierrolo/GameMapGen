# GameMapGen

[agent_gif]: https://github.com/Pierrolo/GameMapGen/blob/main/content/example.gif "Trained Agent"
[ExampleMap]: https://github.com/Pierrolo/GameMapGen/blob/main/content/example_map.png "ExampleMap"
[ExampleMap_2]: https://github.com/Pierrolo/GameMapGen/blob/main/content/example_map_2.png "ExampleMap_2"
[ExampleMap_3]: https://github.com/Pierrolo/GameMapGen/blob/main/content/example_map_3.png "ExampleMap_3"

![Trained Agent][agent_gif]
![ExampleMap][ExampleMap]
![ExampleMap_2][ExampleMap_2]
![ExampleMap_3][ExampleMap_3]

## Description 


DQN applied to grid world map generation, with many different parameters to be changed, using keras with tensorflow backend.
If you want to run this code, you can find [here](https://github.com/Pierrolo/GameMapGen/blob/main/content/environment_conda.yml) the conda env file with all the packages.


Three files to be run :
1. train.py 
2. test.py
3. MakeGifs.py
All three scripts use a config file. 
The config file for train.py is  Config/train_config.config in it several parameters can be set.
As for test.py and MakeGifs.py, a model name must be provided, and the script will load the config file used to train that model automatically.


In the description below, the parameters you can change are written in __bold__.




## Environment
The environments is a square grid world of size (__map_size__) with five types of tiles : empty, wall, start, treasure, finish.
A map is considered playable when there is only one of each Start, Treasure and Finish and a there is a path between Start and Treasure (avoiding walls and finish) and between treasure and finsih. These paths are computed using A* algorithm.


The state is the map, with each cell having the id value of its tile.
The agent can change any tile it wants into any other tile (cf the _wide_ approach in the "PCGRL: Procedural Content Generation via Reinforcement Learning" paper [(arXiv:2001.09212)](https://arxiv.org/abs/2001.09212)). You can disable useless action with parameter __mask_useless_action__, which prevents action that change one tile into itself (e.g. put a wall where there is already a wall).


The reward drives the agent towards playable maps with larger distances between Start and Treasure and between Treasure and Finish.


The episode ends when one of two condition is met. First when the map is playable. Second when the agent has made a certain amount of steps, parameter __max_nb_steps_ratio__ which represents the ratio of the map the agent can change during an episode (e.g. with max_nb_steps_ratio = 0.2 and map_size = 8 the episode will last at most 13 steps).
The reward is discounted using the specified __gamma__ value.

Since DQN poduces a deterministic policy, the generated map is highly dependent on the inital state. Therefore, at the beginning of the episode, the inital state is generated randomly using 2 arguments : wall_ratio and key_elem_ratio. The probability of spawning a key element (start, treasure and finish) is equal to key_elem_ratio/map_size^2 and the rest of the map is filled with wall with the probablity wall_ratio. So high wall_ratio value will generate a initial map with a lot of walls and high key_elem_ratio will generate an intial map with a lot of key elements.


## Algorithm
The algorithm used is DQN. It is trained for a given number of episodes (__EPISODES__ parameter).
The exploration is epsilon-greedy with a starting value of __epsilon__, a decay of  __epsilon_decay__ at the end of each episode and min value of  __epsilon_min__. There are additional parameters, such as the frequency of target network update (__update_target_network_ep__), the number of episodes to run before starting training (__nb_of_warm_up_ep__) and the frequency at which weights are stored (__save_weight_every_nb_ep__). 

The buffer used is a prioritized buffer ([(arXiv:1511.05952)](https://arxiv.org/abs/1511.05952)) of size __capacity__ and of prioritizing value of __alpha__

Finally, there is a possibility to enact an automatic curriculum ([(arXiv:1910.07224)](https://arxiv.org/abs/1910.07224)) (__enable_auto_curriculum__), which select automatically the values of wall_ratio and key_elem_ratio for each episode. Removing it will sample randomly both values in a \[0,1\] interval.




## Neural Network

Because of the sate and the action space have similar shape (state: (map_size * map_size), action: (map_size * map_size * 5)) there are three possibilities of network architecture for it __model_type__.
It can be [_FullyConv_](https://github.com/Pierrolo/GameMapGen/blob/main/content/model_FullyConv.png), _FractalNet_, [_U-net_](https://github.com/Pierrolo/GameMapGen/blob/main/content/model_Unet.png).
Moreover, there is the possibility to apply the dueling Q network architecture ([(arXiv:1511.06581)](https://arxiv.org/abs/1511.06581)) (__dueling_arch__).
The loss function __loss_fn__ can either be mse or huber. The optimizer __opt_type__ can be Adam, SGD, or RMS. And you can change the value (or remove) of the gradient clipping __clipnorm__.


## Training
```
python train.py --config-fn --config-dir
```
You can create your own config file, although it is highly rencommanded you only run ``` python train.py ``` and change the congif file directly. 

During training, a tensorboard file ([examlpe here](https://github.com/Pierrolo/GameMapGen/blob/main/content/tensorboard_example.PNG)) will be generated in "reporting \ __model_name__ \".
Weights will be saved regularly into "models \ __model_name__ \". In addition to the weights, the config file will be saved to be able to use it later for testing. Finally, the python script responsible for building the model is also saved. This is a bit unhordotox, but this allows to easily recreate the same model and apply the saved weights on it. Saving the model architecture directly is not feasible in most cases, as our approach relies on several Lambda keras layers.



## Testing And Reporting
```
python  test.py --model-name --checkpoint-nb --EPISODES
```
```
python  MakeGifs.py --model-name --checkpoint-nb --EPISODES  
```
model-name is the name of the trained model, checkpoint-nb is an int referencing which of the checkpoint to load (if null then it takes the most recent one) and EPISODES the number of episodes to run per combination of reset parameters.

After providing a __model_name__ (we provide an already trained model, on map size = 8, called Example_model) to be tested, when executing the script test.py or MakeGifs.py, the script will fetch the associated model builder and the saved weights specified to re-create the desired agent.
In test.py, the agent is ask to play several episodes with varying lenghts and initial state parameters. subsequently it will produce a a few reporting plots and the best maps it generated.

[image_test_1]: https://github.com/Pierrolo/GameMapGen/blob/main/content/test_results_1.png "test_results_1"
[image_test_2]: https://github.com/Pierrolo/GameMapGen/blob/main/content/test_results_2.png "test_results_2"

![test_results_1][image_test_1] ![test_results_2][image_test_2]

In MakeGifs.py, the agent will play a few number of episodes which will be used to generate a gif. This gif will be saved in "reporting\ __model_name__"; next to the tensorboard file.



## Future Possible Improvements

There are quite a few issues with the current approach and several improvements could me made.


1. DQN is deterministic. This creates quite a few problems. First the generated maps depend highly on th initial state, therefor improving the intial state generation would drastically improve the be performance of the agent.
Second, this deterministic nature make is hard to perform exploration. At the beginning of the training, the first successul maps are quite empty without walls; this create an agent which learns to generate one of each key elem and then somewhat randomly remove walls
To address this, using an actor critic with a probabilistic policy could help. Furthermore, the actor might be encoded differently, for instance, do a softmax beteen each tile and another between eahc position and the product is your policy

2. RL for this type of problem isn't necessarly best suited. A GAN appraoch which would generate teh map in a single step, and then be rewarded like the RL agent, based on the quality of the map, would probably perform better.

3. Improve the state space. For now, the state is just the map. Adding more informations, for instance aggregated info such as the number of each type of tile,wether thare is a path between the key elems and how long is that path, the number of steps left in the episode. All this would help the agent perfomr better

4. Improve the action masking, for instance forbif the agent to change a tile it just changed, or it changed in the last X steps.

5. The tiles are put through an embedding layer, maybe use it also when outputting the action, like output a vector of the same size as the embedding size and look at which of the tiles is closer to it and do that action

6. Giving the fact that the env is very small, it would make sens to run parrallel environment, they won't evne need to be asynchronous.

7. Difficulty assessment. We use A* to find the paths (S to T and T to F). In addition to reporting the path, we might report the number of iteration need to find it, this could be a computation of how difficult the map is, and might be used to orient the reward toward more difficult maps.

8. Add more reporting. During training, compute the q values for each tile type and report it, looking as gif the agent seem to only be a bit better than random...
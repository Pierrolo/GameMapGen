# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 10:38:36 2023

@author: woill
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda

CONV_SIZE = 32

def full_convs(convs_input ):
    
    conv_1 = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(convs_input)
    conv_2 = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_1)
    conv_3 = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_2) #tf.keras.layers.Add()([conv_1, conv_2]))
    conv_4 = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_3)
    conv_5 = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_4)
    conv_6 = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_5)
    conv_7 = tf.keras.layers.Convolution2D(filters = CONV_SIZE//2, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_6)
    
    conv_out = conv_7
    return conv_out


def FractalNet(convs_input ):   
    def fractal_block(inputs_, conv_layer):
        ## Simple fractal block of depth 4
        conv_1 = conv_layer(inputs_)
        conv_2 = conv_layer(conv_1)
        merged_conv = tf.keras.layers.Average()([conv_1, conv_2])
        
        conv_3 = conv_layer(merged_conv)
        conv_4 = conv_layer(conv_3)
        merged_conv = tf.keras.layers.Average()([conv_1, conv_3, conv_4])
        
        
        conv_5 = conv_layer(merged_conv)
        conv_6 = conv_layer(conv_5)
        merged_conv = tf.keras.layers.Average()([conv_5, conv_6])
        
        conv_7 = conv_layer(merged_conv)
        conv_8 = conv_layer(conv_7)
        merged_conv = tf.keras.layers.Average()([conv_1, conv_5, conv_7, conv_8])
        
        return merged_conv
    
    conv_1 = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(convs_input)
    
    conv_layer_2 = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")    
    conv_2 = fractal_block(conv_1, conv_layer_2)
    
    conv_layer_3 = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")    
    conv_3 = fractal_block(conv_2, conv_layer_3)
    
    conv_layer_4 = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")    
    conv_4 = fractal_block(conv_3, conv_layer_4)
    
    conv_layer_5 = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")    
    conv_5 = fractal_block(conv_4, conv_layer_5)
    
    conv_layer_6 = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")    
    conv_6 = fractal_block(conv_5, conv_layer_6)
    
    conv_layer_7 = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")    
    conv_7 = fractal_block(conv_6, conv_layer_7)
    
    conv_out = conv_7
    return conv_out




def get_model_builder(env, args_training):
    
    map_size = env.map_size
    nb_of_elements = len(env.all_elements)
    input_size = env.state_size
    # action_size = env.action_size
    
    if args_training.opt_type == "Adam":
        opt = tf.keras.optimizers.Adam(lr=args_training.learning_rate)
    elif args_training.opt_type == "SGD":
        opt = tf.keras.optimizers.SGD(lr=args_training.learning_rate)
    elif args_training.opt_type == "RMS":
        opt = tf.keras.optimizers.RMSprop(lr=args_training.learning_rate)
        
    embed_size = 4
    
    def model_builder(opt = opt):
        ## Input is a flat vector
        # input_state = tf.keras.Input(shape= (input_size,))
        input_state = tf.keras.Input(shape= (input_size,))
        
        # map_size = Lambda(lambda x : tf.cast(tf.sqrt(tf.cast(tf.shape(x)[1], dtype = tf.float32)), dtype = tf.int32))(input_state)
        
        ## One hot encoding
        one_hot_encoded = Lambda(lambda x: tf.one_hot(tf.cast(x, dtype = tf.uint8), nb_of_elements))(input_state)
        state_reshapped_square_one_hot = tf.keras.layers.Reshape((map_size, map_size, nb_of_elements))(one_hot_encoded)
        
        ## Create 1 channel per id
        embedding_layer = tf.keras.layers.Embedding(nb_of_elements, embed_size)
        input_embedded = embedding_layer(input_state) ## shape = (batch_size, map_size**2, embed size)
        state_embedded_square = tf.keras.layers.Reshape((map_size, map_size, embed_size))(input_embedded)
        
        convs_input = state_embedded_square
        
        if args_training.include_xtra_features:
            ## Compute extra features, like sums and such
            sums_of_elems = Lambda(lambda x: tf.ones_like(x) * tf.math.reduce_sum(x, (1,2), keepdims = True))(state_reshapped_square_one_hot) ## each channel is egal to the number of that element
            convs_input = tf.keras.layers.Concatenate()([state_embedded_square, state_reshapped_square_one_hot, sums_of_elems])
            
        
        ## Apply conv layers        
        if args_training.model_type == "FullyConv":
            conv_out = full_convs(convs_input = convs_input)
        elif args_training.model_type == "FractalNet":
            conv_out = FractalNet(convs_input = convs_input)

        
        if args_training.separate_pos_and_elem :## add softmax layer
            conv_max = tf.keras.layers.MaxPool2D(pool_size = (2,2))(conv_out)
            conv_max = tf.keras.layers.Convolution2D(filters = CONV_SIZE//2, kernel_size = [2,2], strides = (2,2), activation = "relu", padding = "same")(conv_max)
            flatten = tf.keras.layers.Flatten()(conv_max)
            dense = flatten
            for neurons in [CONV_SIZE*2,CONV_SIZE,CONV_SIZE//2]:
                dense = tf.keras.layers.Dense(neurons, activation = "tanh")(dense)
            dense_sftmx = tf.keras.layers.Dense(nb_of_elements, activation = "softmax")(dense)
            
            conv_action = tf.keras.layers.Convolution2D(filters = nb_of_elements, kernel_size = [1,1], strides = (1,1), activation = None, padding = "same", use_bias=False)(conv_out)
            
            conv_action = Lambda(lambda x: x[1] * tf.expand_dims(tf.expand_dims(x[0], axis = 1), axis = 1))((dense_sftmx, conv_action))
            
        else : 
            ## Output as a conv layer still
            conv_action = tf.keras.layers.Convolution2D(filters = nb_of_elements, kernel_size = [1,1], strides = (1,1), activation = None, padding = "same", use_bias=False)(conv_out)
            
        ## Mask useless actions
        if args_training.mask_useless_action : 
            conv_action = Lambda(lambda x: x[0] - x[1]*tf.stop_gradient(tf.math.reduce_max(x[0], (1,2,3), keepdims = True)))((conv_action, state_reshapped_square_one_hot))
            # conv_action - state_reshapped_square_one_hot*max(conv_action)
        
        
        ## Flatten and output actions
        action_flatten = tf.keras.layers.Flatten()(conv_action)
        
        model = Model(inputs=input_state, outputs=action_flatten)
        ## tf.keras.utils.plot_model(model)
        
        model.compile(loss='mse', optimizer=opt)
        
        model.summary()
        
        return model
    
    return model_builder
















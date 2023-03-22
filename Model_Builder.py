# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 10:38:36 2023

@author: woill
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda


CONV_SIZE = 32

delta = 1.0
def huber_loss(y_true, y_pred):
    x = y_true - y_pred
    hub_loss = tf.where(tf.math.less_equal(tf.abs(x), delta), 0.5*tf.square(x), delta*tf.abs(x) -0.5*(delta**2))
    return tf.reduce_mean(hub_loss, axis=-1)  # Note the `axis=-1`

def full_convs(convs_input ):
    
    conv_1 = tf.keras.layers.Convolution2D(filters = CONV_SIZE,    kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(convs_input)
    conv_2 = tf.keras.layers.Convolution2D(filters = CONV_SIZE,    kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_1)
    conv_3 = tf.keras.layers.Convolution2D(filters = CONV_SIZE,    kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_2) #tf.keras.layers.Add()([conv_1, conv_2]))
    conv_4 = tf.keras.layers.Convolution2D(filters = CONV_SIZE,    kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_3)
    conv_5 = tf.keras.layers.Convolution2D(filters = CONV_SIZE,    kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_4)
    conv_6 = tf.keras.layers.Convolution2D(filters = CONV_SIZE,    kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_5)
    conv_7 = tf.keras.layers.Convolution2D(filters = CONV_SIZE//2, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_6)
    
    conv_out = conv_7
    return conv_out


def U_Net(convs_input):
    
    conv_downs = []
    current_conv = convs_input
    for i in range(int(np.log(int(convs_input.shape[1]))/np.log(2)-1)):
        current_conv = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(current_conv)
        current_conv = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (2,2), activation = "relu", padding = "same")(current_conv)
        conv_downs.append(current_conv)
    
    dense = tf.keras.layers.Flatten()(current_conv)
    for i in range(2): 
        dense = tf.keras.layers.Dense(int(dense.shape[1]), activation = "sigmoid")(dense)
    
    conv_up = tf.keras.layers.Reshape((int(current_conv.shape[1]), int(current_conv.shape[2]), CONV_SIZE))(dense)
    for i in range(int(np.log(int(convs_input.shape[1]))/np.log(2)-1)):
        conv_up = tf.keras.layers.Concatenate()([conv_up, conv_downs[-(i+1)]])
        conv_up = tf.keras.layers.Conv2DTranspose(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_up)
        conv_up = tf.keras.layers.Conv2DTranspose(filters = CONV_SIZE, kernel_size = [2,2], strides = (2,2), activation = "relu", padding = "same")(conv_up)
    
    conv_out = conv_up
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
        opt = tf.keras.optimizers.Adam(lr=args_training.learning_rate, clipnorm=args_training.clipnorm)
    elif args_training.opt_type == "SGD":
        opt = tf.keras.optimizers.SGD(lr=args_training.learning_rate, clipnorm=args_training.clipnorm)
    elif args_training.opt_type == "RMS":
        opt = tf.keras.optimizers.RMSprop(lr=args_training.learning_rate, clipnorm=args_training.clipnorm)
        
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
        elif args_training.model_type == "UNet":
            conv_out = U_Net(convs_input = convs_input)  


        if args_training.separate_pos_and_elem :
            
            ## Value function stream
            conv_val = conv_out
            for _ in range(int(np.log(map_size)/np.log(2)-1)) :
                conv_val = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (2,2), activation = "relu", padding = "same")(conv_val)
            dense_val = tf.keras.layers.Flatten()(conv_val)
            for neurons in [CONV_SIZE*2,CONV_SIZE,CONV_SIZE//2]: 
                dense_val = tf.keras.layers.Dense(neurons, activation = "relu")(dense_val)
            v_function = tf.keras.layers.Dense(1, activation = None)(dense_val)
            
            
            ## Advantage function stream
            conv_adv = conv_out
            for _ in range(3) : 
                conv_adv = tf.keras.layers.Convolution2D(filters = CONV_SIZE, kernel_size = [2,2], strides = (1,1), activation = "relu", padding = "same")(conv_adv)
            conv_adv = tf.keras.layers.Convolution2D(filters = nb_of_elements, kernel_size = [1,1], strides = (1,1), activation = None, padding = "same")(conv_adv)
            
            
            
            conv_action = Lambda(lambda x : tf.reshape(x[0], (-1,1,1,1)) + (x[1] - tf.reduce_mean(x[1] * (1-x[2]), axis = (1,2,3), keepdims=True)))((v_function, conv_adv, state_reshapped_square_one_hot))
            """
            if args_training.mask_useless_action : 
                ## make the mean over the advantage over the unmasked actions only
                conv_adv = Lambda(lambda x: x[0] - x[1]*tf.stop_gradient(tf.math.reduce_max(tf.abs(x[0]), (1,2,3), keepdims = True)))((conv_adv, state_reshapped_square_one_hot))
                # norm adv with max
                conv_action = Lambda(lambda x : tf.reshape(x[0], (-1,1,1,1)) + (x[1] - tf.reduce_max(x[1], axis = (1,2,3), keepdims=True)))((v_function, conv_adv))
            else : 
                # norm adv with mean
                conv_action = Lambda(lambda x : tf.reshape(x[0], (-1,1,1,1)) + (x[1] - tf.reduce_mean(x[1], axis = (1,2,3), keepdims=True)))((v_function, conv_adv))
            """

        else : 
            ## Output as a conv layer still
            conv_action = tf.keras.layers.Convolution2D(filters = nb_of_elements, kernel_size = [1,1], strides = (1,1), activation = None, padding = "same", use_bias=False)(conv_out)
            
        ## Mask useless actions
        if args_training.mask_useless_action : 
            conv_action = Lambda(lambda x: x[0] - x[1]*tf.stop_gradient(tf.math.reduce_max(tf.abs(x[0]), (1,2,3), keepdims = True)))((conv_action, state_reshapped_square_one_hot))
            # conv_action - state_reshapped_square_one_hot*max(conv_action)
        
        
        ## Flatten and output actions
        action_flatten = tf.keras.layers.Flatten()(conv_action)
        
        model = Model(inputs=input_state, outputs=action_flatten)
        ## tf.keras.utils.plot_model(model, show_shapes = True)
        
        
        if args_training.loss_fn == "mse" : 
            model.compile(loss='mse', optimizer=opt)
        elif args_training.loss_fn == "huber" : 
            model.compile(loss=huber_loss, optimizer=opt)
        
        
        
        # model.summary()
        
        return model
    
    return model_builder
















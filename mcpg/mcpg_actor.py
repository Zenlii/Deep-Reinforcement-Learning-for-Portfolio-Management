"""
Actor

function 1, make decision: 

    input (s_t), output(action_t)

function 2, update decisition model:

    input (learning_rate, G_t), output(new model)
    
function 3, update the tagrget actor network

"""


import tensorflow as tf
from tensorlayer.layers import (BatchNorm, Conv2d, Dropout , Dense, Flatten, Input, LocalResponseNorm, MaxPool2d)
from tensorlayer.models import Model
import tensorlayer as tl
import numpy as np
import os 


# ===========================
#   Actor
# ===========================


class Actor_NetWork(object):
    '''
    tf_session: tensorflow session
    state_shape: shape of state
    action_shape: 
    learning_rate: learning rate of actor
    target_lr : learning rate of target actor network
    batch size: size of mini batch, used to training
    '''
    
    
    def __init__(self, gamma, state_shape, action_shape, a_learning_rate):
        
        self.gamma = gamma # discount_factor 
        
        self.state_shape = state_shape # should be [None, m_stock, historic_window, feature]
        self.action_shape = action_shape # should be [None, m_stock]
        
        # Acotr Network
        self.actor_learning_rate = a_learning_rate
        self.actor_network = self.get_cnn_actor_model(self.state_shape ,"Actor_Network") # tensorlayer model
        self.actor_network.train()
        self.actor_opt = tf.optimizers.Adam(self.actor_learning_rate)
  

        
  
    
    def Generate_action(self, states, greedy = False):
        '''
        states shape should be [m_stock, historic_window, feature]
        greedy is used to determine random explore 
        
        '''
        
        if greedy:
            new_action = self.actor_network(states)
            return new_action
        else:
            new_action = self.actor_network(states) 
            new_action = new_action + np.random.normal(0, 0.01, np.shape(new_action))
            new_action = np.clip(new_action, 0 ,1) # values outside the interval are clipped to the interval edges
            new_action = new_action/np.sum(new_action)
            new_action = np.array(new_action).astype(np.float32)
            return new_action 
    
    def learn(self, states, Gt):
        '''
        inputs: (states_t,actions_t,rewards_t,states_t+1)
        inputs shape: [[batch_size, m_stock, historic_window, feature], [batch_size, actions] \
            ,[batch_size], [[batch_size, m_stock, historic_window, feature]]
            
        used to update network
        
        '''

        #G_t must be discount to time 0
        G_t = tf.constant(Gt, dtype=tf.float32)
        
        # actor gradients - Monte Carlo Policy Gradient
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.actor_network.trainable_weights)
            actions = - G_t * tf.math.log(self.actor_network(states)) # theta <- theta + b * Gt * gradient
        actor_grads =  tape.gradient(actions, self.actor_network.trainable_weights)
        #print(actor_grads)
        
        #num_of_layer = int((len(actor_grads) + 1)/2)
        #for layer in range(0, num_of_layer): # for 11 layers
            #print('max gradient of layer={}, kernel={}, bias={}'.format( \
            #layer, actor_grads[layer].numpy().max(), actor_grads[layer*2+1].numpy().max()))
        
        # update actor 
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor_network.trainable_weights))
        

           
    def save(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join(["MCPG", "PM"]))
        if not os.path.exists(path):
            os.makedirs(path) # create a new dir
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor_network)

    def load(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join(["MCPG", "PM"]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor_network)
    

    def get_cnn_actor_model(self, inputs_shape, model_name):
        # self defined initialization
        stock_num = inputs_shape[1]
        his_window = inputs_shape[2]
        feature_num = inputs_shape[3]
        W_init = tl.initializers.truncated_normal(stddev=1e-1)
        W_init2 = tl.initializers.truncated_normal(stddev=1e-1)
        b_init2 = tl.initializers.constant(value=5e-13)

        # build network
        ni = Input(inputs_shape)
        nn = Conv2d(feature_num, (1, 1), (1, 1), padding='VALID', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv1')(ni) #fully connected
        #nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(nn)
        nn = Conv2d(feature_num, (1, his_window), (1, 1), padding='VALID', act=tf.nn.relu, W_init=W_init2, b_init=None, name='conv2')(nn)
        nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(nn)
        
        nn = Flatten(name='flatten')(nn)
        #nn = Dropout(keep=0.5)(nn)
        #nn = Dense(stock_num , act=tf.nn.relu, name='dense1relu')(nn) #  W_init=W_init2, b_init=b_init2,
        #nn = BatchNorm()(nn)
        #nn = Dense(32, act=tf.nn.relu, name='dense2relu')(nn) # W_init=W_init2, b_init=b_init2,
        #nn = BatchNorm()(nn)
        nn = Dense(stock_num, act=tf.nn.softmax,  name='output')(nn) # W_init=W_init2,
        M = Model(inputs=ni, outputs=nn) # , name=model_name
        return M
           
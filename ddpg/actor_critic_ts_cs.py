"""
Actor

function 1, make decision: 

    input (s_t), output(action_t)

function 2, update decisition model:

    input (learning_rate, Q_gradiant(at sample s_i, a_i)), output(new model)
    
function 3, update the tagrget actor network

"""


import tensorflow as tf
from tensorlayer.layers import (BatchNorm, Conv2d, Dropout , Dense, Flatten, Input, LocalResponseNorm, MaxPool2d, Transpose)
from tensorlayer.models import Model
import tensorlayer as tl
import numpy as np
import os 

# ===========================
#   Actor & Critic
# ===========================


class Actor_Critic_NetWork(object):
    '''
    tf_session: tensorflow session
    state_shape: shape of state
    action_shape: 
    learning_rate: learning rate of actor
    target_lr : learning rate of target actor network
    batch size: size of mini batch, used to training
    '''
    
    
    def __init__(self, gamma, state_shape, action_shape, a_learning_rate, target_lr, c_learning_rate):
        
        self.gamma = gamma # discount_factor 
        self.target_lr = target_lr # target network learning rate
        
        self.state_shape = state_shape # should be [None, m_stock, historic_window, feature]
        self.action_shape = action_shape # should be [None, m_stock]
        
        # Acotr Network
        self.actor_learning_rate = a_learning_rate
        self.actor_network = self.get_cnn_actor_model(self.state_shape ,"Actor_Network") # tensorlayer model
        self.actor_network.train()
        self.target_actor_network = self.get_cnn_actor_model(self.state_shape, "Target_Actor_Network")
        self.target_actor_network.train()
        self.actor_opt = tf.optimizers.Adam(self.actor_learning_rate)
        self.initial_target_network(self.actor_network, self.target_actor_network)
        
        # Critic Network
        self.Critic_learning_rate = c_learning_rate        
        self.critic_network = self.get_cnn_critic_model(self.state_shape, 
                                                        self.action_shape,"Critic_Network") # tensorlayer model
        self.critic_network.train()
        self.target_critic_network = self.get_cnn_critic_model(self.state_shape,
                                                               self.action_shape,"Target_Critic_Network")
        self.target_critic_network.train()
        self.critic_opt = tf.optimizers.Adam(self.Critic_learning_rate)
        self.initial_target_network(self.critic_network, self.target_critic_network)
        
    def initial_target_network(self, from_model, to_model):
        '''
        Let the target newwork parameter same as actor/critic
        
        '''
        
        for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)
        

        
    def update_target(self):
        """
        Soft updating by exponential smoothing
        
        Update the target net work para
        
        """
        paras = self.actor_network.trainable_weights + self.critic_network.trainable_weights
        paras_target = self.target_actor_network.trainable_weights + self.target_critic_network.trainable_weights
        for i, j in zip(paras_target, paras):
            i.assign(tf.multiply(j, self.target_lr) + tf.multiply(i, (1 - self.target_lr))) # soft update
    
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
            new_action = new_action + np.random.normal(0, 0.1, np.shape(new_action))
            new_action = np.clip(new_action, 0 ,1) # values outside the interval are clipped to the interval edges
            new_action = new_action/np.sum(new_action)
            new_action = np.array(new_action).astype(np.float32)
            return new_action 
    
    def learn(self, inputs):
        '''
        inputs: (states_t,actions_t,rewards_t,states_t+1)
        inputs shape: [[batch_size, m_stock, historic_window, feature], [batch_size, actions] \
            ,[batch_size], [[batch_size, m_stock, historic_window, feature]]
            
        used to update network
        
        '''
        states = inputs[0]
        actions =  inputs[1]
        rewards = inputs[2]
        next_states = inputs[3]
        
        # critic gradients
        with tf.GradientTape() as tape:
            actions_ = self.target_actor_network(next_states)
            q_ = self.target_critic_network([next_states, actions_])
            y = rewards + self.gamma * q_
            q = self.critic_network([states, actions])
            td_error = tf.losses.mean_squared_error(y, q)
        critic_grads = tape.gradient(td_error, self.critic_network.trainable_weights)
        
        # actor gradients
        with tf.GradientTape() as tape:
            actions = self.actor_network(states)
            q = self.critic_network([states, actions])
            actor_loss = -tf.reduce_mean(q)  # maximize the q
        actor_grads = tape.gradient(actor_loss, self.actor_network.trainable_weights)
        
        # update actor and critic
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic_network.trainable_weights))  
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor_network.trainable_weights))
        
        # update the target network
        self.update_target()
           
    def save(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join(["DDPG_cs", "PM"]))
        if not os.path.exists(path):
            os.makedirs(path) # create a new dir
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor_network)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor_target.hdf5'), self.target_actor_network)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic_network)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic_target.hdf5'), self.target_critic_network)

    def load(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join(["DDPG_cs", "PM"]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor_network)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor_target.hdf5'), self.target_actor_network)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic_network)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic_target.hdf5'), self.target_critic_network)

    def get_cnn_actor_model(self, inputs_shape, model_name):
        # self defined initialization
        stock_num = inputs_shape[1]
        his_window = inputs_shape[2]
        feature_num = inputs_shape[3]
        W_init = tl.initializers.truncated_normal(stddev=5e-5)
        W_init2 = tl.initializers.truncated_normal(stddev=5e-5)
        b_init2 = tl.initializers.constant(value=5e-13)

        # build network
        ni = Input(inputs_shape)
        nn = Conv2d(feature_num , (1, 1), (1, 1), padding='SAME', act=tf.nn.relu, \
                    W_init=W_init, b_init=None, in_channels = feature_num,  name='conv1')(ni) 
        #nn = Dropout(keep=0.5)(nn)
        nn = MaxPool2d((1, 3), (1, 3), padding='SAME', name='pool1')(nn)
        #fully connected
        
        #nn = Conv2d(1, (1, 1), (1, 1),\
                    #padding='VALID', act=tf.nn.relu, \
                    #in_channels = feature_num ,W_init=W_init, b_init=None, name='conv2')(nn)
                    
        
                    
         
        #nn = Dropout(keep=0.5)(nn)
        #nn = Conv2d(1, (1, 1), (1, 1),\
                    #padding='VALID', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv3')(nn)
        

        #nn = Conv2d(feature_num, (1, 1), (1, 1), padding='SAME', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv2')(nn)
        #nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(nn)

        nn = Flatten(name='flatten')(nn)
        #nn = BatchNorm()(nn)
        #nn = Dropout(keep=0.5)(nn)
        
        #nn = BatchNorm()(nn)
        #nn = Dense(20, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='dense2relu')(nn)
        #nn = BatchNorm()(nn)
        nn = Dense(stock_num, act=tf.nn.softmax, W_init=W_init2, name='output')(nn)
        
        
        #nn2 = Transpose(perm=[0, 2, 1, 3], conjugate=False, name='trans')(ni)
        nn2 = Conv2d(stock_num , (stock_num, 1), (1, 1), padding='SAME', act=tf.nn.relu, \
                    W_init=W_init, b_init=None, data_format =  'channels_first',  name='conv1_1')(ni) 
        nn2 = MaxPool2d((1, 3), (1, 3), padding='VALID', name='pool1_1')(nn2)
        
        nn2 = Flatten(name='flatten1')(nn2)
        #nn2 = Dense(50, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='dense1relu1')(nn2)
        nn2 = Dense(stock_num, act=tf.nn.softmax, W_init=W_init2, name='output1')(nn2)
        
        nn = tl.layers.Concat(1)([nn, nn2])
        nn = Dense(50, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='dense1relu')(nn)
        nn = Dense(stock_num, act=tf.nn.softmax, W_init=W_init2, name='output3')(nn)
        
        M = Model(inputs=ni, outputs=nn) # , name=model_name
        return M
           
    def get_cnn_critic_model(self, state_shape, action_shape, model_name):
        stock_num = action_shape[1]
        his_window = state_shape[2]
        feature_num = state_shape[3]

        # self defined initialization
        W_init = tl.initializers.truncated_normal(stddev=5e-5)
        W_init2 = tl.initializers.truncated_normal(stddev=5e-5)
        b_init2 = tl.initializers.constant(value= 5e-13)

        # build network

        state_input = tl.layers.Input(state_shape, name='s_input')
        action_input = tl.layers.Input(action_shape, name='a_input')
        nn = Conv2d(feature_num  , (1, 1), (1, 1), padding='SAME',\
                    in_channels = feature_num, act=tf.nn.relu, W_init=W_init, b_init=None, name='conv1')(state_input) 
        #fully connected
        #nn = Conv2d((feature_num - 1), (1, his_window -2), (1, 1),\
                    #padding='VALID', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv2')(nn)
        #nn = Conv2d(1, (1, 1), (1, 1),\
                    #padding='VALID', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv3')(nn)
        nn = MaxPool2d((1, 3), (1, 3), padding='SAME', name='pool1')(nn)

        nn = Conv2d(1, (1 , 1), (1, 1), padding='VALID', act=tf.nn.relu,  W_init=W_init, b_init=None, \
                    in_channels = feature_num, name='conv3')(nn)
        #nn = Dropout(keep=0.5)(nn)
        #nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(nn)

        nn = Flatten(name='flatten')(nn)
        #nn = Dropout(keep=0.5)(nn)
        #nn = Dense(320, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='dense1relu')(nn)
        #nn = BatchNorm()(nn)
        nn = Dense(stock_num, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='dense2relu')(nn)


        nn = tl.layers.Concat(1)([nn, action_input])
        nn = Dense(60, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='dense3relu')(nn)
        #nn = BatchNorm()(nn)
        nn = Dense(1, W_init=W_init, b_init=b_init2 , name='output')(nn)
        M = Model(inputs=[state_input, action_input], outputs=nn) # , name=model_name
        return M
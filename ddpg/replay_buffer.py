# Notice that shape of reward should be [n,1]
# Shape of action should be [n, m], where m is the stock size 




import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed = 123):
        """

        """
        self.buffer_size = buffer_size
        # we need to create different buffer coz they are different shape
        self.clear()
        random.seed(random_seed)

    def add(self, s, a, r, s2):
        '''
        inputs must has shape [1, ...]
        as critic network output has shape=(2, 1)
        rewards = np.array([[1],[2]])
        '''
        index = self.count % self.buffer_size
        self.buffer_s[index] = s
        self.buffer_a[index] = a
        self.buffer_r[index] = r
        self.buffer_s2[index] = s2
        self.count += 1

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            index = list(np.arange(0, self.count, 1))
            batch = random.sample(index, self.count)
        else:
            size = min(self.buffer_size,self.count,batch_size)
            index = list(np.arange(0, size, 1))
            batch = random.sample(index, size)

        s_batch = self.buffer_s[batch]
        a_batch = self.buffer_a[batch]
        r_batch = self.buffer_r[batch]
        s2_batch = self.buffer_s2[batch]
        
        inputs = [self.batch_reshape(s_batch), self.batch_reshape(a_batch), 
                  r_batch.reshape(len(r_batch),1) ,self.batch_reshape(s2_batch) ]

        return inputs
    
    def batch_reshape(self, batch):
        '''
        used to reshape the batch to fit network input
        This step may be slow, we will update it later !
        
        '''
        a = batch[0]
        if len(batch) == 1:
            return a
        else:
            for i in batch[1:len(batch)]:
                a = np.append(a, i ,axis = 0)
            return a
        

    def clear(self):
        self.buffer_s = np.array([None] * self.buffer_size)
        self.buffer_a = np.array([None] * self.buffer_size)
        self.buffer_r = np.array([None] * self.buffer_size)
        self.buffer_s2 = np.array([None] * self.buffer_size)
        self.count = 0
from __future__ import print_function

from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
import gym.spaces

from datetime import datetime, timedelta

eps = 1e-10



def sharpe(returns, freq=252, rf=0):
    return (np.sqrt(freq) * np.mean(returns - rf + eps)) / np.std(returns - rf + eps)


def max_drawdown(returns):
    """ Max drawdown """
    log_r = np.log(1 + returns)
    log_cum_r = np.cumsum(log_r)
    r_box = log_cum_r.copy()
    for i in range(len(returns)):
        r_box[i] = log_cum_r[i] - np.max(log_cum_r[0:i])
    MD = 1 - np.exp(np.min(r_box))
    
    return MD

def dataframe_to_numpy(data, index_name, shape):
    '''
    change multiindex dataframe to numpy array
    
    '''
    a = data[index_name[0]].to_numpy().reshape(shape)
    a = a.astype(np.float32)
    for i in index_name[1:]:
        b = data[i].to_numpy().reshape(shape)
        b = b.astype(np.float32)
        a = np.append(a,b, axis = 0)
    return a

def update_weight(w0, r0):
    if sum(r0 * w0) != 0:
        dw0 = (r0 * w0) / sum(r0 * w0)
    else:
        dw0 = w0 * 0 # keep the size 
    return dw0
        

class DataGenerator(object):
    """Acts as data provider for each new episode."""

    def __init__(self, history, abbreviation, steps = 200, window_length = 5, eps_move = 10, start_date = None):
        """
        Args:
            history:  MultiIndex pandas DataFrame with shape 
            (his_window, num_stocks * feature) 
                feature: open, high, low, close, volume
            abbreviation: stock name
            steps: the total number of steps to simulate, default is 200 days
            window_length: observation window
            eps_move: move the start date at each reset in roll
            start_date: Start Date
        """
        import copy
        
        
        self.reset_pointer = 0
        
        self.step = 0
        self.steps = steps 
        self.window_length = window_length
        
        
        self.num_stock = len(abbreviation)
        self.num_feature = int(history.shape[1]/self.num_stock)
        
        self.eps_move = eps_move
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d' ) # e.g., "2017-10-22"

        # make immutable class
        self._data = history.copy()  # all data
        self.asset_names = copy.copy(abbreviation)

    def _step(self):
        # get observation matrix from history

        self.step += 1
        
        obs = self.data[:, self.step:self.step + self.window_length, :].copy()
        obs = obs.reshape(1, self.num_stock, self.window_length, self.num_feature)
        obs = obs.astype(np.float32)


        # used for compute optimal action and sanity check
        ground_truth_obs = self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()

        done = self.step  >= self.steps 
        return obs, done, ground_truth_obs

    def reset(self):
        self.step = 0
        self.reset_pointer += 1

        # get data for this episode, each episode might be different.
        if self.start_date is None:
            self.idx = np.random.randint(
                low=self.window_length, high=self._data.shape[1] - self.steps)
        else:
            # compute index corresponding to start_date for repeatable sequence
            self.idx = self.start_date + timedelta(days=(self.reset_pointer - 1) * self.eps_move)
            
            
        # data start with self.idx - self.window_length
        # find  start date - window size 
        
        start_date_windows = self._data.loc[:self.idx].copy() 
        # all the history before the start date (include the start date, by loc method)
        start_date_windows = start_date_windows.iloc[(-self.window_length ),:] # so that start date 
        start_date_windows = str(start_date_windows.name)[0:10] # start date - window size, 
        # we cant just let date - timedelta(days = window size) since the weekend and holiday are not count in data but count in timedelta
        
        data = self._data.loc[start_date_windows:].copy()
        data = data.iloc[0:(self.window_length + self.steps + 2) , :] # +2 for true_growth
        assert data.shape[0] > 0, \
                'Invalid start date, must be window_length day after start date and simulation steps day before end date'
        
        # transform the data to numpy array with shape (m_stock, his, features)
        data = dataframe_to_numpy(data, self.asset_names, (1, data.shape[0], self.num_feature))
        self.data = data
        
        #  first obs
        obs = data[:, self.step:self.step + self.window_length, :]
        obs = obs.reshape(1, self.num_stock, self.window_length, self.num_feature)
        obs = obs.astype(np.float32)
        
        return obs, \
               self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()
    
    
def best_performance_stock(y):
    '''
    y have form (1.1 , 1.2, 0.9, ...) which is the past return
    '''
    loc = np.argmax(y)
    w = np.zeros(len(y))
    w[loc] = 1
    return w

def performance_rank(y):
    '''
    return the rank of return, e.g. [1.1,1.3,0.9] -> [1,2,0] best performance give highest value 
    
    '''
    x = y.argsort()
    ranks = np.empty_like(x)
    ranks[x] = np.arange(len(y))
    return ranks


class PortfolioSim(object):
    """
    Compute the reward and record the step
    
    """

    def __init__(self, asset_names=list(), steps=200, trading_cost=0.0025, time_cost=0.0, alpha = 0, \
                 beta = 0 ,gamma_ = 0.01):
        self.asset_names = asset_names
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.reset()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma_ = gamma_

    def _step(self, w1, w0, observation, ground_truth_obs):
        """
        Used to compute rewards based on given action and observation
        
        Args:
            w1 - new action of portfolio weights - e.g. [[0.1,0.9,0.0]] coz its output of network
            w0 - previous action 
            y0 - previous price relative vector, also called return
                e.g. [1.0, 0.9, 1.1]
            observation used to compute reward, has shape (1, m_stock, his_window, features)
            beta: for variance
            gamma: for max weight
            
        w0 = 0 for the initial states
        w1 will be the first weight
        
        """
        alpha = self.alpha
        beta = self.beta
        gamma_ = self.gamma_
      
        w1 = w1[0] # e.g., [[0.1,0.9,0.0]]
        w0 = w0[0]
        
        # assert sum(w1) != 1.0, 'weight sum are not equal to 1'
        #if sum(w1) != 1.0:
            #print(sum(w1))
        
        num_stock = len(w1)

        p0 = self.p0 # portfolio value
        
        if sum(w0) == 0: # initial step
            dw0 = w0
            previous_return = 1
            y0 = np.array([1] * num_stock)
            
            # equal weights portfolio 
            equal_hold_weight = np.array([0] * num_stock)
            update_equal_hold_weight = np.array([1/num_stock] * num_stock)
            eq_r = 1
            
            # best performance stocks
            window_past_return = np.array(observation[0][:, -2, 3]/observation[0][:, 0, 3])
            bp_weight = np.array([0] * num_stock)
            update_bp_weight = best_performance_stock(window_past_return)
            self.bp_weight = bp_weight
            bp_r = 1
            
            self.weighted_rank = 4.5
            
            
        else:
            close_price_vector = observation[0][:, -1, 3] # obs has shape (1, m_stock, window, feature)
            previous_close_price_vector = observation[0][:, -2, 3]
            y0 = np.array(close_price_vector / previous_close_price_vector)
            previous_return = np.dot(y0, w0)
            dw0 = update_weight(w0, y0)  # update past weight
            
            
            # equal weights portfolio 
            equal_hold_weight = np.array([1/num_stock] * num_stock)
            update_equal_hold_weight = update_weight(equal_hold_weight, y0)
            eq_r = np.dot(equal_hold_weight, y0) # period return 
            
            
            # best performance stocks
            x = observation[0][:, -2, 3]/observation[0][:, 0, 3]
            window_past_return = np.array(x)
            update_bp_weight = best_performance_stock(window_past_return)
            bp_r = np.dot(update_bp_weight, y0)
            
            # track the weighted rank 
            #x_ = ground_truth_obs[0][:, 3]/observation[0][:, -1, 3]
            self.weighted_rank = np.dot(performance_rank(y0) , w0)
            
        
        # equal weights portfolio 
        equal_hold_weight_cost = self.cost * np.sum(np.abs(equal_hold_weight - update_equal_hold_weight))
        eq_r = eq_r * (1 - equal_hold_weight_cost) # add transaction cost 
        eq_r = eq_r * (1 - self.time_cost)
        self.eq_p0 = self.eq_p0 * eq_r
        
        
        
        # best past performance stocks, weight has form [0,0,1,....,0]
        bp_weight_cost = self.cost * np.sum(np.abs(update_bp_weight - self.bp_weight))
        bp_r = bp_r * (1 - bp_weight_cost) # add transaction cost 
        bp_r = bp_r * (1 - self.time_cost)
        self.bp_p0 = self.bp_p0 * bp_r
        self.bp_weight = update_bp_weight # update the weight
        
        # network portfolio 
        mu1 = self.cost * np.sum((np.abs(w1 - dw0))) 

        assert mu1 < 1.0, f'trading cost is too large: {mu1}'
                        
        p1 = p0 * (1 - mu1) * previous_return  # update final portfolio value
        rho1 = p1/p0 - 1
        r1 = np.log(1 + rho1)  # log rate of return
        
        
        # predicted variance of portfolio
        z = np.cov(observation[0][:,:,3], rowvar = True)
        predicted_var = np.dot(np.matmul(w1 ,z), w1)
        
        # max of weight
        log_max_w1 =  max(w1)
        
        # log eqr
        log_eq_r = np.log(eq_r)
        
        
        reward = (r1  - alpha *  log_eq_r - beta * predicted_var - gamma_ * log_max_w1 )
        # remember for next step
        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        done = (p1 <= 0)

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "average_return": np.mean(y0),
            "rate_of_return": rho1,
            "weights_std": np.std(w1),
            "cost": mu1,
            'equal_weight_portfolio_value': self.eq_p0,
            'MOM_portfolio_value': self.bp_p0,
            'portfolio_rank_weight': self.weighted_rank
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.p0 = 1.0
        self.eq_p0 = 1.0 # equal weight prortfolio value
        self.bp_p0 = 1.0 # track best past stock given time window
        self.weighted_rank = 4.5 # initial weighted rank, equal to equal weight portfolio
        
def observation_normalized(observation, num_stock, window_length):
    # normalize the open, high, low, close by divided the last close
    d1 = observation[:,:,:,0:4]/observation[:,:,-1,3].reshape(1,num_stock,1,1) # 
    # normalize the vol
    d2 = observation[:,:,:,4]/observation[:,:,-1,4].reshape(num_stock,1) 
    d2 = d2.reshape((1,num_stock,window_length,1))
    d = np.concatenate([d1,d2],axis = 3)
    return d
    
    


class PortfolioEnv(gym.Env):
    """
    Rl environment for PM
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self,
                 history,
                 abbreviation,
                 steps = 200,  # 2 years
                 trading_cost = 0.0025,
                 time_cost = 0.0,
                 window_length = 5,
                 eps_move = 10,
                 sample_start_date=None,
                 alpha = 0, 
                 beta = 0,
                 gamma_ = 0.01
                 ):
        """
        An environment for financial portfolio management.
        Params:
            steps - steps in episode
            trading_cost 
            window_length - length of past observations 
            eps_move - move the start date ar each rest
            sample_start_date - start date 
        """
        self.window_length = window_length
        self.num_stocks = len(abbreviation)
        self.trading_cost = trading_cost

        self.src = DataGenerator(history, abbreviation, steps=steps, window_length=window_length, eps_move=eps_move,
                                 start_date=sample_start_date)
        
        self.date = history

        self.sim = PortfolioSim(
            asset_names=abbreviation,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps,
            alpha = alpha, 
            beta = beta,
            gamma_ = gamma_
            )
        
        # store the previous action
        self.previous_action = np.array([0] * len(self.src.asset_names)).reshape(1 , len(self.src.asset_names))
        self.previous_action = self.previous_action.astype(np.float32) 

        # openai gym attributes
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Box(
            0, 1, shape=(1, len(self.src.asset_names)), dtype=np.float32)  # exclude cash

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, len(abbreviation), window_length,
                                                                                 history.shape[-1]), dtype=np.float32)
        
    def step(self, action):
        return self._step(action)
    

    def _step(self, action):
        """
        Step the env.
        Actions should be portfolio [[w0...]]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """

        reward, info, done2 = self.sim._step(action, self.previous_action,\
                                             self.previous_observation, self.previous_ground_truth_obs ) # compute the reward
        
        # add dates
        info['date'] = self.date_track[self.src.step] #self.start_idx + timedelta(days = self.src.step)
        # current step
        info['steps'] = self.src.step
        info['next_obs'] = self.previous_ground_truth_obs
        
        observation, done1, ground_truth_obs = self.src._step() # move 1 step 
        self.previous_observation = observation # update for next round
        self.previous_ground_truth_obs = ground_truth_obs
        

 
        
        # normalized the data up to the last close(for open, high, low, close) and last volume (only for vol)
        obs_norm = observation_normalized(observation, self.num_stocks, self.window_length)
        
        # update the information and action
        self.infos.append(info)   
        self.previous_action = action 

        return obs_norm, reward, done1 or done2, info
    
    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        observation, ground_truth_obs = self.src.reset()
        self.start_idx = self.src.idx
        self.date_track = self.date.loc[self.start_idx:].index # track the true date
        self.previous_observation = observation # compute the reward, no need for norm
        self.previous_ground_truth_obs = ground_truth_obs
        
        obs_norm = observation_normalized(observation, self.num_stocks, self.window_length)
        
        # reset the previous action
        self.previous_action = np.array([0] * len(self.src.asset_names)).reshape(1 , len(self.src.asset_names))
        self.previous_action = self.previous_action.astype(np.float32) 
        
        info = {}
        info['next_obs'] = ground_truth_obs
        return obs_norm, info

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'human':
            self.plot()
            
    def render(self, mode='human', close=False):
        return self._render(mode='human', close=False)

    def plot(self):
        # show a plot of portfolio, equal weighted portfolio, and simple MOM
        df_info = pd.DataFrame(self.infos)
        # print(df_info)
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        mdd = max_drawdown(df_info.rate_of_return)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sharpe_ratio)
        df_info[["portfolio_value", "equal_weight_portfolio_value", 'MOM_portfolio_value']].plot(title=title, fig=plt.gcf(), rot=30)
    
    def table(self):
        # show a plot of portfolio, equal weighted portfolio, and simple MOM
        df_info = pd.DataFrame(self.infos)
        # print(df_info)
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        
        # RL portfolio
        mdd = max_drawdown(df_info.rate_of_return)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        win_pec= len(df_info.rate_of_return[df_info.rate_of_return > 0]) / len(df_info.rate_of_return)
        annual_return = df_info.portfolio_value[-1] ** (252/len(df_info.portfolio_value))
        print(f" Sharpe Ratio = {sharpe_ratio}")
        print(f" MDD = {mdd}")
        print(f" Winning percentage = {win_pec}")
        print(f" Annual Return = {annual_return}")
        
        return [sharpe_ratio, mdd, win_pec, annual_return]
        
        
        
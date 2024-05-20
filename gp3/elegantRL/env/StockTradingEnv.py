import gymnasium as gym
import numpy as np
from numpy import random as rd
import torch


class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        turbulence_thresh=99,
        max_stock=None,
        min_stock_rate=0.25,
        initial_capital=1e5,
        initial_stocks=None,
        gpu_id: int = 0,
        num_envs=1,
    ):
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        turbulence_ary = config["turbulence_array"]
        gamma = config.get("gamma", 0.99)
        reward_scaling = config.get("reward_scaling", 2**-11)
        if_train = config["if_train"]
        
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}" if (torch.cuda.is_available() and (self.gpu_id >= 0)) else "cpu")
        self.num_envs = num_envs
        
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary

        self.tech_ary = self.tech_ary * 2**-7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (
            self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        cash_dim = 1
        turbulence_dim = 1
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )
        
        # reset()
        self.current_step = None
        self.cash = None
        self.stocks = None
        self.total_assets = None
        self.gamma_reward = None
        self.initial_total_assets = None
        self.cumulative_return = None
        
        # environment information
        self.env_name = "StockTradingEnv"
        self.state_dim = (
            cash_dim
            + turbulence_dim
            + stock_dim
            + stock_dim
            + stock_dim
            + self.tech_ary.shape[1]
        )
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        price = self.price_ary[self.current_step]
        self.cumulative_return = 0
        
        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 51, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.cash = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.cash = self.initial_capital
        
        self.rewards = []
        self.total_assets = self.cash + (self.stocks * price).sum()
        self.initial_total_assets = self.total_assets
        self.gamma_reward = 0.0
        observation = self.get_state(price)
        info = {}
        return observation

    def step(self, action):
        self.current_step += 1
        price = self.price_ary[self.current_step]
        self.max_stock = np.round(np.floor(100_000 / price)).astype(int)
        action = np.round((action * self.max_stock)).astype(int)
        min_action = np.round((self.max_stock * self.min_stock_rate)).astype(int)
        self.stocks_cool_down += 1
        
        if self.turbulence_bool[self.current_step] == 0:
            
            # Sell Logic
            for index in np.where((action < -min_action) & (self.stocks_cool_down > 25))[0]:
                if price[index] > 0:
                    sell_num_shares = min(self.stocks[index], -action[index])
                    sell_value = price[index] * sell_num_shares
                    self.stocks[index] -= sell_num_shares
                    self.cash += sell_value
                    self.stocks_cool_down[index] = 0

            # Buy Logic
            for index in np.where((action > min_action) & (self.stocks_cool_down > 25))[0]:
                if price[index] > 0:
                    buy_num_shares = min(self.cash // price[index], action[index])
                    buy_value = price[index] * buy_num_shares
                    self.stocks[index] += buy_num_shares
                    self.cash -= buy_value
                    self.stocks_cool_down[index] = 0

        # turbulence logic
        else:
            self.cash += (self.stocks * price).sum()
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0
        
        # Reward Calculations
        observation = self.get_state(price)
        total_assets = self.cash + (self.stocks * price).sum()
        reward = (total_assets - self.total_assets) * self.reward_scaling
        self.rewards.append(reward)
        self.total_assets = total_assets
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        terminated = self.current_step == self.max_step
        truncated = False
        info = {}
        if terminated:
            reward = self.gamma_reward
            self.cumulative_return = total_assets / self.initial_total_assets

        return observation, reward, terminated, info

    def get_state(self, price):
        cash = np.array(self.cash * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        observation = np.hstack(
            (
                cash,
                self.turbulence_ary[self.current_step],
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_ary[self.current_step],
            )
        )
        # print(len(observation))
        return observation

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
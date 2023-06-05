from collections import defaultdict #allow access to keys
import matplotlib.pyplot as plt #plots
#from matplotlib.patches import patch #draw shapes
import numpy as np #data and array manipulation
import seaborn as sns

from tqdm import tqdm #progress bar

import gymnasium as gym

env = gym.make('Blackjack-v1', sab=True, render_mode='rgb_array')

class BlackjackAgent:
    
    def __init__(
        self,
        learning_rate:float,
        initial_epsilon:float,
        epsilon_decay:float,
        final_epsilon:float,
        discount_factor:float=0.95
    ):
        # Initialise RL agent with empty dictionary of state-action value (Q-value), a learning rate and an epsilon
        # discount_factor: the discount factor for computing the Q-value
        self.q_values = defaultdict(lambda:np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs):
        #Epsilon greedy 
        #Best action based on current policy p = 1-e
        #Random action p = e
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            #Q-value function is used to estimate the optimal action to take in each state
            return int(np.argmax(self.q_values[obs]))

    # Update the Q-value of actions
    def update(self, obs, action, reward, terminated, next_obs):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * temporal_difference
        self.training_error.append(temporal_difference)
        
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

#observation, info = env.reset()

#observation = (16, 9, False)
#observation is tuple of: 
# 1. Player current sum
# 2. Value of dealer card
# 3. Boolean - useable ace without bust

#sample a random action - training loop
#action = env.action_space.sample()

#execute the action
#observation, reward, terminated, truncated, info = env.step(action)

#observation=(24,10,False)
#reward=-1.0
#terminated=True
#truncated=False
#info={}

#if terminated:
#    env.reset()

#Blackjack - policy learned over time by updating the action-value estimates 
#of state-action pairs based on rewards received during the game
#Game played many times to learn the optimal strategy

#hyperparameters
learning_rate = 0.01
n_episodes = 100000
initial_epsilon = 1.0
epsilon_decay = initial_epsilon/ (n_episodes/2) #reduce the exploration rate over time
final_epsilon = 0.1

agent = BlackjackAgent(
    learning_rate = learning_rate,
    initial_epsilon = initial_epsilon,
    epsilon_decay = epsilon_decay,
    final_epsilon = final_epsilon
)

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play with one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(obs, action, reward, terminated, next_obs)
        frame = env.render()
        plt.show()

        done = terminated or truncated
        obs = next_obs
    
    agent.decay_epsilon()


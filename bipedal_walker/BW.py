from collections import deque
import random
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn
import torch.nn.functional as F

env = gym.make("BipedalWalker-v3")
MEMORY = deque(maxlen=10000)
EPSILON = 1  # exploratory action probability
GAMMA = 1  # future reward discounting
SAMPLES = []
EPSILON_DECAY = 0.999
N_EPISODES = 20000
BATCH_SIZE = 256
ACTION_SPACE = 4
STATE_FEATURE_DIMENSIONALITY = 6 * 4
REWARD_THRESHOLD_TO_SAVE = -10000
TAU = 1e-2


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


actor = Actor(STATE_FEATURE_DIMENSIONALITY, 8, ACTION_SPACE)
actor_target = Actor(STATE_FEATURE_DIMENSIONALITY, 8, ACTION_SPACE)
critic = Critic(STATE_FEATURE_DIMENSIONALITY + ACTION_SPACE, 8, ACTION_SPACE)
critic_target = Critic(STATE_FEATURE_DIMENSIONALITY + ACTION_SPACE, 8, ACTION_SPACE)

# We initialize the target networks as copies of the original networks
for target_param, param in zip(actor_target.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(critic_target.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)

critic_criterion = torch.nn.MSELoss()
critic_optimizer = torch.optim.Adam(critic.parameters())
actor_optimizer = torch.optim.Adam(actor.parameters())

total_reward_per_episode = []
epsilon_per_episode = []
for e in range(N_EPISODES):
    with torch.no_grad():
        EPSILON = EPSILON * EPSILON_DECAY
        state = torch.tensor(env.reset(), dtype=torch.float)
        done = False
        i = 0
        total_reward = 0
        while not done:
            if random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = MODEL(state.unsqueeze(0)).squeeze(0)

            next_state, reward, done, _ = env.step(action)
            # env.render()
            next_state = torch.tensor(next_state, dtype=torch.float)
            MEMORY.append((state, action, reward, next_state, done))
            state = next_state
            i += 1
            total_reward += reward
        total_reward_per_episode.append(total_reward)
        epsilon_per_episode.append(EPSILON * 100)
        if e % 100 == 0:
            plt.clf()
            plt.plot(total_reward_per_episode, linestyle="None", marker='.')
            plt.plot(epsilon_per_episode)
            plt.pause(interval=0.0001)

        x_batch = torch.zeros((BATCH_SIZE, STATE_FEATURE_DIMENSIONALITY))
        y_batch = torch.zeros((BATCH_SIZE, ACTION_SPACE))

        minibatch = random.sample(MEMORY, min(len(MEMORY), BATCH_SIZE))
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            y_batch[i] = MODEL(state)
            y_batch[i, action] = reward if done else reward + GAMMA * max(MODEL(next_state))
            x_batch[i] = state

        q_vals = critic(state, action)
        next_action = actor_target(next_state)
        next_q = critic_target(next_state, next_action.detach())
        q_prime = reward + GAMMA * next_q
        critic_loss = critic_criterion(q_vals, q_prime)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        for target_param, param in zip(actor_target.parameters(), actor.parameters()):
            target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))

        for target_param, param in zip(critic_target.parameters(), critic.parameters()):
            target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))


plt.clf()
plt.plot(total_reward_per_episode, linestyle="None", marker='.')
plt.plot(epsilon_per_episode)
plt.pause(interval=0.0001)
plt.show()

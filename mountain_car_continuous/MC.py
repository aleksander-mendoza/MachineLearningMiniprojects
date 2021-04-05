import sys
from collections import deque
import random
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn
import torch.nn.functional as F
import select

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

env = gym.make("MountainCarContinuous-v0")
MEMORY = deque(maxlen=10000)
EPSILON = 1  # exploratory action probability
GAMMA = 1  # future reward discounting
SAMPLES = []
EPSILON_DECAY = 0.99
N_EPISODES = 20000
BATCH_SIZE = 1024
ACTIONS_NUM = env.action_space.shape[0]  # 4
STATES_NUM = env.observation_space.shape[0]   # 6 * 4
REWARD_THRESHOLD_TO_SAVE = -90
TAU = 1e-2


class OUNoise(object):
    def __init__(self, action_space):
        self.action_dim = action_space.shape[0]
        self.low = torch.from_numpy(action_space.low)
        self.high = torch.from_numpy(action_space.high)
        self.reset()

    def reset(self):
        self.state = torch.zeros(self.action_dim)

    def get_action(self, action):
        self.state = (1 - 0.15) * self.state + EPSILON * torch.randn(self.action_dim)
        return torch.min(torch.max(action + self.state, self.low), self.high)


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
        x = torch.cat((state, action), dim=1)
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


actor = Actor(STATES_NUM, 4, ACTIONS_NUM).to(DEVICE)
actor_target = Actor(STATES_NUM, 4, ACTIONS_NUM).to(DEVICE)
critic = Critic(STATES_NUM + ACTIONS_NUM, 4, 1).to(DEVICE)
critic_target = Critic(STATES_NUM + ACTIONS_NUM, 4, 1).to(DEVICE)

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


def train():
    reward_batch = torch.zeros(BATCH_SIZE)
    state_batch = torch.zeros((BATCH_SIZE, STATES_NUM))
    action_batch = torch.zeros((BATCH_SIZE, ACTIONS_NUM))
    next_state_batch = torch.zeros((BATCH_SIZE, STATES_NUM))
    done_batch = torch.zeros((BATCH_SIZE, STATES_NUM))

    minibatch = random.sample(MEMORY, min(len(MEMORY), BATCH_SIZE))
    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
        next_state_batch[i] = next_state
        # y_batch[i, action] = reward if done else reward + GAMMA * max(MODEL(next_state))
        state_batch[i] = state
        action_batch[i] = action
        reward_batch[i] = reward
        done_batch[i] = done
    next_state_batch = next_state_batch.to(DEVICE)
    action_batch = action_batch.to(DEVICE)
    state_batch = state_batch.to(DEVICE)
    done_batch = done_batch.to(DEVICE)
    reward_batch = reward_batch.unsqueeze(1).to(DEVICE)
    q_vals = critic(state_batch, action_batch)
    next_action = actor_target(next_state_batch)
    next_q = critic_target(next_state_batch, next_action.detach())
    q_prime = reward_batch + GAMMA * next_q * (1 - done_batch)
    critic_loss = critic_criterion(q_vals, q_prime)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    policy_loss = -critic.forward(state_batch, actor.forward(state_batch)).mean()

    actor_optimizer.zero_grad()
    policy_loss.backward()
    actor_optimizer.step()

    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))

    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))


print("type anything to toggle rendering")
noise = OUNoise(env.action_space)
visualize = False
for e in range(N_EPISODES):

    i, _, _ = select.select([sys.stdin], [], [], 0)
    if i:
        cmd = sys.stdin.readline().strip()
        visualize = not visualize

    EPSILON = EPSILON * EPSILON_DECAY
    state = torch.tensor(env.reset(), dtype=torch.float)
    done = False
    i = 0
    total_reward = 0
    rewards = []
    while not done:
        with torch.no_grad():
            action = actor(state.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
        action = noise.get_action(action)
        next_state, reward, done, _ = env.step(action)
        if visualize:
            env.render()
        next_state = torch.tensor(next_state, dtype=torch.float)
        rewards.append((state, action, reward, next_state, done))
        state = next_state
        i += 1
        total_reward += reward
    if total_reward >= REWARD_THRESHOLD_TO_SAVE:
        MEMORY += rewards
    total_reward_per_episode.append(total_reward)
    epsilon_per_episode.append(EPSILON * 100)
    if len(MEMORY) >= BATCH_SIZE:
        train()
    if e % 1 == 0:
        plt.clf()
        plt.plot(total_reward_per_episode, linestyle="None", marker='.')
        plt.plot(epsilon_per_episode)
        plt.pause(interval=0.0001)

plt.clf()
plt.plot(total_reward_per_episode, linestyle="None", marker='.')
plt.plot(epsilon_per_episode)
plt.pause(interval=0.0001)
plt.show()

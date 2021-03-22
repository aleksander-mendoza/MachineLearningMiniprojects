from collections import deque
import random
import matplotlib.pyplot as plt
import gym
import torch

env = gym.make("CartPole-v1")
MEMORY = deque(maxlen=10000)
EPSILON = 1  # exploratory action probability
GAMMA = 1  # future reward discounting
SAMPLES = []
EPSILON_DECAY = 0.999
N_EPISODES = 10000
BATCH_SIZE = 256
STATE_FEATURE_DIMENSIONALITY = 4


class DQN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear4 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # torch.relu()
        h_relu1 = self.linear1(x).clamp(min=0)
        h_relu2 = self.linear2(h_relu1).clamp(min=0)
        h_relu3 = self.linear3(h_relu2).clamp(min=0)
        y_pred = self.linear4(h_relu3)
        return y_pred


MODEL = DQN(STATE_FEATURE_DIMENSIONALITY, 16, 2)
LOSS_CRITERION = torch.nn.MSELoss()
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), weight_decay=0.01)

# episode = []
# observation = torch.tensor(env.reset())
# for _ in range(1000):
#     env.render()
#
#     observation, reward, done, info = env.step(action)
#     observation = torch.tensor(observation)
#     if done:
#         observation = torch.tensor(env.reset())


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
                action = torch.argmax(MODEL(state.unsqueeze(0))).item()

            next_state, reward, done, _ = env.step(action)
            # env.render()
            next_state = torch.tensor(next_state, dtype=torch.float)
            MEMORY.append((state, action, reward, next_state, done))
            state = next_state
            i += 1
            total_reward += reward
        total_reward_per_episode.append(total_reward)
        epsilon_per_episode.append(EPSILON*100)
        plt.clf()
        plt.plot(total_reward_per_episode)
        plt.plot(epsilon_per_episode)
        plt.pause(interval=0.0001)

        if i < 499:
            x_batch = torch.zeros((BATCH_SIZE, STATE_FEATURE_DIMENSIONALITY))
            y_batch = torch.zeros((BATCH_SIZE, 2))

            minibatch = random.sample(MEMORY, min(len(MEMORY), BATCH_SIZE))
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                y_batch[i] = MODEL(state)
                y_batch[i, action] = reward if done else reward + GAMMA * max(MODEL(next_state))
                x_batch[i] = state
    if i < 499:
        predictions = MODEL(x_batch)
        loss = LOSS_CRITERION(predictions, y_batch)
        loss.backward()
        OPTIMIZER.step()
        OPTIMIZER.zero_grad()

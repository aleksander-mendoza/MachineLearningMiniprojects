from collections import deque
import random
import matplotlib.pyplot as plt
import gym
import torch

env = gym.make("Acrobot-v1")
# env = gym.make("CartPole-v1") # it also works perfectly well with CartPole
MEMORY = deque(maxlen=10000)
EPSILON = 1  # exploratory action probability
GAMMA = 1  # future reward discounting
SAMPLES = []
EPSILON_DECAY = 0.999
N_EPISODES = 20000
BATCH_SIZE = 256
ACTION_SPACE = 3  # change to 2 if running CartPole
STATE_FEATURE_DIMENSIONALITY = 6  # Change to 4 if running CartPole
REWARD_THRESHOLD_TO_SAVE = -500  # Change to something between 0 and 400 if running CartPole


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


MODEL = DQN(STATE_FEATURE_DIMENSIONALITY, 16, ACTION_SPACE)
LOSS_CRITERION = torch.nn.MSELoss()
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), weight_decay=0.01)

total_reward_per_episode = []
epsilon_per_episode = []
for e in range(N_EPISODES):
    with torch.no_grad():
        EPSILON = EPSILON * EPSILON_DECAY
        state = torch.tensor(env.reset(), dtype=torch.float)
        done = False
        i = 0
        total_reward = 0
        TMP_MEMORY = []
        while not done:
            if random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = torch.argmax(MODEL(state.unsqueeze(0))).item()

            next_state, reward, done, _ = env.step(action)
            # env.render()
            next_state = torch.tensor(next_state, dtype=torch.float)
            TMP_MEMORY.append((state, action, reward, next_state, done))
            state = next_state
            i += 1
            total_reward += reward
        if total_reward > REWARD_THRESHOLD_TO_SAVE:
            MEMORY += TMP_MEMORY
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
    predictions = MODEL(x_batch)
    loss = LOSS_CRITERION(predictions, y_batch)
    loss.backward()
    OPTIMIZER.step()
    OPTIMIZER.zero_grad()

plt.clf()
plt.plot(total_reward_per_episode, linestyle="None", marker='.')
plt.plot(epsilon_per_episode)
plt.pause(interval=0.0001)
plt.show()

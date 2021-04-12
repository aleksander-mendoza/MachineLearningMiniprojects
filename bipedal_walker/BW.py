from collections import deque
import random
import matplotlib.pyplot as plt
import gym
import torch

env = gym.make("BipedalWalker-v3")
# env = gym.make("CartPole-v1") # it also works perfectly well with CartPole
MEMORY = deque(maxlen=10000)
EPSILON = 1  # exploratory action probability
GAMMA = 1  # future reward discounting
SAMPLES = []
EPSILON_DECAY = 0.999
N_EPISODES = 20000
BATCH_SIZE = 256

state = torch.tensor(env.reset(), dtype=torch.float)
while True:
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    env.render()
    state = torch.tensor(state, dtype=torch.float)

exit()


class Dreamer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(Dreamer, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        self.inverse_model = torch.nn.Sequential(
            torch.nn.Linear(input_size * 2, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_size)
        )
        self.forward_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size + action_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        self.action_embedding = torch.nn.Linear(in_features=action_size, out_features=hidden_size)
        self.recurrent = torch.nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.imagined_latent = torch.randn(hidden_size, dtype=torch.float, requires_grad=False)
        self.cell = torch.randn((1, hidden_size), dtype=torch.float, requires_grad=False)
        self.previous_latent = torch.randn(hidden_size, dtype=torch.float, requires_grad=False)
        self.last_action = torch.randn(action_size, dtype=torch.float, requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=0.01, lr=0.001)
        self.criterion = torch.nn.MSELoss()

    def forward(self, curr_state):
        latent_state = self.features(curr_state)
        inverse_action = self.inverse_model(torch.cat((self.previous_latent, latent_state)))
        next_action = self.actor(self.imagined_latent)

        action_embedding = torch.relu(self.action_embedding(next_action))
        next_predicted_latent, self.cell = self.recurrent(action_embedding.unsqueeze(0),
                                                          (self.imagined_latent.unsqueeze(0), self.cell))
        next_predicted_latent = next_predicted_latent.squeeze(0)

        loss_between_latents = self.criterion(self.imagined_latent, latent_state)

        loss_between_actions = self.criterion(self.last_action, inverse_action)

        novelty_reward = loss_between_latents.item()

        advantage = novelty_reward + GAMMA*self.critic(self.imagined_latent) - self.critic(self.previous_latent)

        total_loss = loss_between_actions + loss_between_latents + advantage
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.imagined_latent.detach_()

        self.imagined_latent = next_predicted_latent
        self.previous_latent = latent_state
        self.last_action = next_action
        return next_action


MODEL = Dreamer(env.observation_space.shape[0], env.observation_space.shape[0], env.action_space.shape[0])

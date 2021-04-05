import copy
import queue
import time
from collections import deque
import random
from math import sqrt

import matplotlib.pyplot as plt
import gym
import torch
from tqdm.auto import tqdm
import threading
POPULATION_SIZE = 1024
MIN_CHANGE_THRESHOLD = 0.001
CHANGE_RATE = 0.001
ELITE_SIZE = 32
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 5
QUEUE = queue.Queue()
ENV_NAME = "BipedalWalker-v3"
TMP_ENV = gym.make(ENV_NAME)
STATE_NUM = TMP_ENV.observation_space.shape[0]
ACTION_NUM = TMP_ENV.action_space.shape[0]
del TMP_ENV
BAR = tqdm(total=POPULATION_SIZE)


class Agent:
    def __init__(self, seq):
        self.seq = seq
        self.fitness = 0


class Worker:
    def __init__(self):
        self.env = gym.make(ENV_NAME)

    def fitness(self, agent):
        total_reward = 0
        state = torch.tensor(self.env.reset(), dtype=torch.float, device=DEVICE)
        done = False
        while not done:
            action = agent(state.unsqueeze(0)).squeeze(0).cpu()
            state, reward, done, _ = self.env.step(action)
            # env.render()
            state = torch.tensor(state, dtype=torch.float, device=DEVICE)
            total_reward += reward
        return total_reward


def busy_loop():
    worker = Worker()
    while QUEUE is not None:
        agent: Agent = QUEUE.get()
        agent.fitness = worker.fitness(agent.seq)
        BAR.update()
        QUEUE.task_done()


for i in range(NUM_WORKERS):
    threading.Thread(target=busy_loop, name="env " + str(i)).start()


def mk_agent(input_size, hidden_size, output_size):
    seq = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(hidden_size, output_size)
    )
    seq.requires_grad_(False)
    seq.to(DEVICE)
    return seq


def copy_agent_and_perturb(source_agent, destination_agent):
    for src, dst in zip(source_agent.parameters(), destination_agent.parameters()):
        dst.data.copy_(src.data)
        perturbation = torch.randn_like(dst)
        perturbation.mul_(abs(perturbation) > 0.9)
        perturbation.mul_(CHANGE_RATE)
        dst.add_(perturbation)


population = [Agent(mk_agent(STATE_NUM, 4, ACTION_NUM)) for _ in range(POPULATION_SIZE)]
copy_agent_and_perturb(population[0].seq, population[1].seq)
torch.set_grad_enabled(False)
mean_fitness = []
max_fitness = []
plus_std_fitness = []
minus_std_fitness = []
for _ in range(1000):
    futures = []
    BAR.reset()
    for a in population:
        QUEUE.put(a)
    QUEUE.join()
    population.sort(key=lambda a: a.fitness)
    elite = population[POPULATION_SIZE - ELITE_SIZE:]
    for i in range(0, POPULATION_SIZE - ELITE_SIZE):
        copy_agent_and_perturb(elite[random.randint(0, ELITE_SIZE)-1].seq, population[i].seq)
    mean = sum(map(lambda x: x.fitness, population)) / POPULATION_SIZE
    mean_fitness.append(mean)
    std = sqrt(sum(map(lambda x: (x.fitness - mean) ** 2, population)) / POPULATION_SIZE)
    max_fitness.append(max(map(lambda x: x.fitness, population)))
    plus_std_fitness.append(mean + std)
    minus_std_fitness.append(mean - std)

    plt.clf()
    plt.plot(mean_fitness, label="mean")
    plt.plot(plus_std_fitness, label="+std")
    plt.plot(minus_std_fitness, label="-std")
    plt.plot(max_fitness, label="max")
    plt.legend(loc='upper left')
    plt.pause(interval=0.0001)

import os

os.putenv("JAVA_HOME", "/Library/Java/JavaVirtualMachines/jdk1.8.0_91.jdk/Contents/Home")

import minerl
import gym
import logging

logging.basicConfig(level=logging.DEBUG)

env = gym.make("MineRLTreechop-v0")

obs = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

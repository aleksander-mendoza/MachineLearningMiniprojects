import os
# Put here your JAVA_HOME path
os.putenv("JAVA_HOME", "/usr/lib/jvm/java-8-openjdk-amd64")

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

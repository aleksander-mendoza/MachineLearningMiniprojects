import os
import time
from collections import OrderedDict
from minerl.env.core import MineRLEnv
import numpy as np
# Put here your JAVA_HOME path
os.putenv("JAVA_HOME", "/usr/lib/jvm/java-8-openjdk-amd64")

import minerl
import gym


class MyActionSpace(gym.spaces.Dict):
    def __init__(self):
        super().__init__(spaces={
            "move": gym.spaces.Discrete(2),
            "camera": gym.spaces.Box(low=-180, high=180, shape=(2,), dtype=np.float32)
        })

    def no_op(self):
        action = OrderedDict()
        action['move'] = 0
        action['camera'] = [0, 0]
        return action


class MyObservationSpace(gym.spaces.Dict):
    def __init__(self):
        super().__init__(spaces={
            'pov': gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
        })

    def no_op(self):
        action = OrderedDict()
        action['pov'] = np.zeros(shape=(84, 84, 3)).astype(int)
        return action


class MyEnvSpec:
    def __init__(self):
        self._action_space = MyActionSpace()
        self._observation_space = MyObservationSpace()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space


xml = "./lava_maze_minerl.xml"
env = minerl.env.core.MineRLEnv(
    xml,
    MyObservationSpace(),
    MyActionSpace(),
    MyEnvSpec()
)

obs = env.reset()

print('Starting Action ...')

# Right-turn around 360 degree
action = env.action_space.no_op()
action['camera'] = [0, 90]
obs, reward, done, info = env.step(action)
time.sleep(1)
obs, reward, done, info = env.step(action)
time.sleep(1)
obs, reward, done, info = env.step(action)
time.sleep(1)
obs, reward, done, info = env.step(action)
time.sleep(1)

# Move forward 1 step
action = env.action_space.no_op()
action['move'] = 1
obs, reward, done, info = env.step(action)
time.sleep(1)

print('Finish')
time.sleep(2)

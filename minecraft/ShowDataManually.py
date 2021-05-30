import os
import minerl
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import random

# minerl.data.download('data') # <-- uncomment to download the dataset


task = 'MineRLObtainDiamond-v0'
recordings = os.listdir('data/' + task)
random.shuffle(recordings)

for recording in recordings:
    observations = np.load('data/' + task + '/' + recording + '/rendered.npz')
    rewards = observations['reward']
    video = cv2.VideoCapture('data/' + task + '/' + recording + '/recording.mp4')
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fc = 0
    ret = True
    while fc < frame_count and ret:
        ret, frame = video.read()
        reward = rewards[fc]
        print(reward)
        fc += 1
        # Normally frames have dimension (64,64,3)
        frame = cv2.resize(frame, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        plt.clf()
        plt.imshow(frame)
        plt.pause(interval=0.001)

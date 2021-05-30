import os
import minerl
import matplotlib.pyplot as plt
import cv2

# minerl.data.download('data') # <-- uncomment to download the dataset

data = minerl.data.make('MineRLObtainDiamond-v0', data_dir='data', minimum_size_to_dequeue=1, num_workers=1)

for current_state, action, reward, next_state, done in data.batch_iter(batch_size=1, preload_buffer_size=1,
                                                                       num_epochs=1, seq_len=32):
    for video_in_batch in current_state['pov']:
        for frame in video_in_batch:
            # Normally frames have dimension (64,64,3)
            frame = cv2.resize(frame, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            plt.clf()
            plt.imshow(frame)
            plt.pause(interval=0.001)
    # Print the POV @ the first step of the sequence
    print(current_state['pov'][0])

    # Print the final reward pf the sequence!
    print(reward[-1])

    # Check if final (next_state) is terminal.
    print(done[-1])

    # ... do something with the data.
    print("At the end of trajectories the length "
          "can be < max_sequence_len", len(reward))

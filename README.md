# Machine Learning Mini-projects

## Solved

- [english phonetics](./phonetics) - seq2seq model for translating english words to their phonetic transcription
   - [(LSTM) Long-Short-Term-Memory units solution](./phonetics/PH.py)
   - [(ResNetLSTM) custom Long-Short-Term-Memory with residual connections, allowing for building deep networks](./phonetics/PH_ResNetLSTM.py)
   - [(CNN-ED) convolutional neural network for building embeddings based on Edit Distance between phonetic transcriptions](./phonetics/CNN.py)
- [acrobot](./acrobot) -  reinforcement learning agent that learns to swing a double pendulum upwards  
   - [(DQN) Deep Q Learning solution](./acrobot/AB.py)
- [MNIST autoencoder](./mnist_autoencoder) - convolutional autoencoder that compresses MNIST digits 
   - [(AE) Autoencoder solution](./mnist_autoencoder/MAE.py)
   - [(VAE) Variational autoencoder solution](./mnist_autoencoder/VAE.py)
- [mountain car](./mountain_car_continuous) -  reinforcement learning agent that learns to drive a car uphill  
   - [(DDPG) Deep deterministic policy gradient solution](./mountain_car_continuous/MC.py)
   - [(PPO) Proximal policy optimisation solution](./mountain_car_continuous/PPO.py) (copied from https://github.com/nikhilbarhate99/PPO-PyTorch)
- [bipedal walker](./bipedal_walker) -  reinforcement learning agent that learns to walk on two feet (on flat terrain)  
   - [(DDPG) Deep deterministic policy gradient solution](./bipedal_walker/DDPG.py) (unfortunately the results are somewhat underwhelming)
   - [(Dreamer) reinforcement learning with latent imagination](./bipedal_walker/Dreamer.py) 
   - [(custom Dreamer) my own adaptation of Dreamer](./bipedal_walker/BW.py) 
   - [(GA) Genetic algorithm](./bipedal_walker/GA.py) (unfortunately the results are poor)
  
        

## Yet to be solved
- [Minecraft exploratory reinforcement learning](./minecraft) - reinforcement learning agent that learns to navigate minecraft world 


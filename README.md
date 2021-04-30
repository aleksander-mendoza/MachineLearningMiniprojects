# Machine Learning Mini-projects

## Solved

- [english phonetics](./phonetics) - seq2seq model for translating english words to their phonetic transcription
   - [(LSTM) Long-Short-Term-Memory units](./phonetics/PH.py)
   - [(ResNetLSTM) custom Long-Short-Term-Memory with residual connections, allowing for building deep networks](./phonetics/PH_ResNetLSTM.py)
   - [(CNN-ED) convolutional neural network for building embeddings based on Edit Distance between phonetic transcriptions](./phonetics/CNN.py)
- [acrobot](./acrobot) -  reinforcement learning agent that learns to swing a double pendulum upwards  
   - [(DQN) Deep Q Learning](./acrobot/AB.py)
- [MNIST autoencoder](./mnist_autoencoder) - convolutional autoencoder that compresses MNIST digits 
   - [(AE) Autoencoder solution](./mnist_autoencoder/MAE.py)
   - [(VAE) Variational autoencoder](./mnist_autoencoder/VAE.py)
   - [(GAN) Generative adversarial neural network](./mnist_autoencoder/GAN.py)
   - [(DCGAN) Deep convolutional generative adversarial neural network](./mnist_autoencoder/DCGAN.py)
- [Moving MNIST autoencoder](./mnist_autoencoder) - State-space model for encoding time series over latent space 
   - [(ConvLSTM) Convolutional LSTM](./moving_mnist_autoencoder/ConvLSTM.py)
   - [(Deterministic LSTM) recurrent network over latent space (with epsilon teacher forcing) ](./moving_mnist_autoencoder/LSTM.py)
   - [(RNN) deterministic recurrent network (without LSTM or any other form of long term memory) over latent space (with epsilon teacher forcing)](./moving_mnist_autoencoder/RNN.py)
   - [(SSM) stochastic state model - variational recurrent network over latent space (with epsilon teacher forcing)](./moving_mnist_autoencoder/SSM.py)
   - [(RSSM) recurrent state space model - recurrent network over latent space with both variational and deterministic connections (with epsilon teacher forcing)](./moving_mnist_autoencoder/SSM.py)
- [mountain car](./mountain_car_continuous) -  reinforcement learning agent that learns to drive a car uphill  
   - [(DDPG) Deep deterministic policy gradient](./mountain_car_continuous/MC.py)
   - [(PPO) Proximal policy optimisation](./mountain_car_continuous/PPO.py) (copied from https://github.com/nikhilbarhate99/PPO-PyTorch)
- [bipedal walker](./bipedal_walker) -  reinforcement learning agent that learns to walk on two feet (on flat terrain)  
   - [(DDPG) Deep deterministic policy gradient](./bipedal_walker/DDPG.py) (unfortunately the results are somewhat underwhelming)
   - [(Dreamer) reinforcement learning with latent imagination](./bipedal_walker/Dreamer.py) 
   - [(custom Dreamer) my own adaptation of Dreamer](./bipedal_walker/BW.py) 
   - [(GA) Genetic algorithm (with elite and no crossover)](./bipedal_walker/GA.py) (the results are poor)
   - [(GA) Genetic algorithm (with crossover and more relaxed survivorship policy)](./bipedal_walker/GA_crossover.py)
- [information extraction from NDA documents](./named_entity_recognition) -  named entity recognition and information extraction tasks  
   - [(BERT) solution based on huggingface pretrained BERT transformers](./named_entity_recognition/BERT.py)
  
        

## Yet to be solved
- [Minecraft exploratory reinforcement learning](./minecraft) - reinforcement learning agent that learns to navigate minecraft world 


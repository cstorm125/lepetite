# lepetite
Implementation of N2N learning: Network to Network Compression via Policy Gradient Reinforcement Learning

* `L12.1 ppo network compression.ipynb` - PPO-Clip with Bernoulli actions to perform network compression
* `L12.2 ppo cartpole.ipynb` - PPO-Clip to play `CartPole-v1`
* `L12.3 framework comparison.ipynb` - PPO-Clip from [@seungeunrho/minimalRL](https://github.com/seungeunrho/minimalRL/blob/master/ppo.py) 
* `L12.4 multi-armed bandits.ipynb` - Multi-armed bandits from [@cstorm125/michael](https://github.com/cstorm125/michael)

With all the hardware accelerations available to us today, many researchers are resorting to larger and larger models to solve their problems. However, larger models also mean more memory usage, longer training and inference time, and more limitations in productionization (for instance, [AWS Lambda](https://docs.aws.amazon.com/lambda/latest/dg/limits.html) only allows 512MB in temporary storage).

Techniques like [knowledge distillation](https://arxiv.org/abs/1503.02531) and [pruning](https://arxiv.org/abs/1611.06440) have been used to reduce the parameters of neural networks. Here we will adapt a layer removal technique from the paper [N2N Learning: Network to Network Compression via Policy Gradient Reinforcement Learning](https://arxiv.org/pdf/1709.06030.pdf) (See codes [here](https://github.com/anubhavashok/N2N)) to compress a VGG-like network to achieve 5x compression and comparable accuracy on [Fashion-MNIST dataset](https://www.kaggle.com/zalando-research/fashionmnist/kernels).

![layer removal](https://i.ibb.co/pXJTM3n/Screen-Shot-2562-11-13-at-23-11-42.png)

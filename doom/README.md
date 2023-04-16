# Doom

The AI learns to play a simplified version of Doom thanks to Deep Convolutional Q-Learning.

![demo](demo.gif)

For the game environment, it uses [gym](https://github.com/openai/gym).

For the model, it uses the Neural Network module from PyTorch, more precisely :
- 3 convolutions
- the ReLU rectifier function
- the Adam optimizer
- the softmax function for probability distribution
- the zero_grad function for stochastic gradient descent
- the MSELoss function for loss function
- a custom Experience Replay memory
- an implementation of Eligibility Trace based on [Asynchronous n-step Q-learning](https://arxiv.org/pdf/1602.01783.pdf) but synchronous

# AI for Doom



# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym

# from gym.wrappers import SkipWrapper
# from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
# import vizdoomgym
from gym import wrappers

# Importing the other Python files
import experience_replay, image_preprocessing



# Part 1 - Building the AI

# Making the brain

class CNN(nn.Module):
    
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        # 3 convolutional layers + 1 hidden layer so 2 full connections:
        # 1 from convoluted and flattened vector to the hidden layer
        # and 1 from the hidden layer to the output layer
        
        # in_channels = 1 for black and white images, 3 for color
        # out_channels = nbr of features we want to detect
        # kernel_size = dimension of window scanning the image (2x2 or 3x3 or 5x5)
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        # in_channels take the outputs of convolution1
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        # in_features = nbr of pixels in the flattened vector, only depends on the images dimension which is 256x256 (x1 because black and white)
        # out_features = nbr of neurons in the hidden layer
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 256, 256)), out_features = 40)
        # in_features take the outputs of fc1
        # out_features should map to the possible actions
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)

    def count_neurons(self, image_dim):
        # create a fake image with random pixels, * is used to convert a triple to list of args
        x = Variable(torch.rand(1, *image_dim))
        # propagate the fake image in the CNN up to fc1: apply convolution + max pooling + relu rectifier
        # max pooling takes kernel size (scanning window size) and strides (by how many pixels it is sliding)
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        # flatten data as a 1 column vector and return its length
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        # propagate the real images to the CNN
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        # flattening layer: why not same as above?
        x = x.view(x.size(0), -1)
        # relu recitfier to break linearity + full connection
        x = F.relu(self.fc1(x))
        # why no relu here?
        x = self.fc2(x)
        return x

# Making the body

class SoftmaxBody(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        # temperature: control the exploration (high temperature = less exploration)
        self.T = T

    def forward(self, outputs):
        # propagate the outputs of the CNN to the action decision function
        probs = F.softmax(outputs * self.T) # distribution of probabilities for each q values
        actions = probs.multinomial() # sample actions according to probs distribution
        return actions

# Making the AI

class AI:

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, input_images):
        # convert input images into numpy array into torch tensor into torch variable
        inputs = Variable(torch.from_numpy(np.array(input_images, dtype = np.float32)))
        outputs = self.brain(inputs)
        actions = self.body(outputs)
        return actions.data.numpy() # return actions as numpy array



# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment
# use image_preprocessing.py to preprocess images from the gym environnement for Doom (again 1x256x256 dimension)

# doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 256, height = 256, grayscale = True)
# doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True) # generate videos of the AI playing Doom
doom_env = image_preprocessing.PreprocessImage(gym.make('VizdoomCorridor-v0'), width = 256, height = 256, grayscale = True)
doom_env = wrappers.Monitor(doom_env, "videos", force = True)

number_actions = doom_env.action_space.n # set of actions for Doom

# Building an AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn, body = softmax_body)

# Setting up Experience Replay
# experience replay is based on eligibility trace technique thus learning by batch of n=10 steps
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10) # accumulate targets and rewards for batch of n steps
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000) # memorize last 10000 steps (split into batches of n steps)


# Implementing Eligibility Trace
# based on Asynchronous n-step Q-learning but with only one agent (so not asynchronous), see https://arxiv.org/pdf/1602.01783.pdf
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    # 1 serie = 10 transitions
    for series in batch:
        # put state of first transition and state of last transition in a numpy array into a torch tensor into a torch variable
        current_input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        current_output = cnn(current_input)
        # cumulative reward is 0 if we reached the last transition of the serie, else the maximum of q values
        cumul_reward = 0.0 if series[-1].done else current_output[1].data.max()
        # iterate from penultimate step to first step
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward # multiply current cumulative reward by gamma + current step reward
        state = series[0].state # state of first transition
        target = current_output[0].data # q value of the input state of the first transition
        target[series[0].action] = cumul_reward # set the target for specific action of first step to cumulative reward
        # only append first input state and first target because learning happends 10 steps afterwards
        inputs.append(state)
        targets.append(target)
    # return inputs as numpy array into torch tensor and targets as torch stack
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

# Making the moving average on n steps
class MA:
    def __init__(self, n):
        self.list_of_rewards = []
        self.size = n
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    def average(self):
        return np.mean(self.list_of_rewards)

# Training the AI
ma = MA(100)
loss = nn.MSELoss() # mean square error loss function
optimizer = optim.Adam(cnn.parameters(), lr = 0.001) # Adam optimizer with learning rate 0.001
nb_epochs = 20 # rerun the training 20 times
for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200)
    # 200 runs of 10 steps by batch of 128 series of 10 steps
    for batch in memory.sample_batch(128):
        # convert inputs and targets to torch variables which use eligibility trace
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn(inputs) # predict using our CNN
        loss_error = loss(predictions, targets) # compute loss error
        optimizer.zero_grad() # init stochastic gradient descent
        loss_error.backward() # backpropagation
        optimizer.step() # update weights using stochastic gradient descent
    # print average reward after each epoch
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))

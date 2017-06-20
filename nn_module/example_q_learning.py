# -*- coding: utf-8 -*-

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

# matplotlib setup
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import diplay


# date type without CUDA
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor

# replay memory
from replay_memory import ReplayMemory, Transition

# q-network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
# utility for reading images
from util import get_screen


def example():
    # unwrapped; to remove time limit
    env = gym.make("CartPole-v0").unwrapped
    plt.ion()

    env.reset()
    plt.figure()
    plt.imshow(get_screen(env).cpu().squeeze(0).permute(1, 2, 0).numpy(),
               interpolation="none")
    plt.title("Example extracted screen")
    plt.show()
    plt.close()


class DQN_agent(object):
    batch_size = 32
    gamma = 0.999
    eps_s = 0.9
    eps_e = 0.05
    eps_d = 200
    memory_size = 10000


    def __init__(self):
        self.last_sync = 0
        self.steps_done = 0
        self.model = DQN()
        # self.model_target = deepcopy(self.model)
        self.optim = optim.RMSprop(self.model.parameters())
        self.memory = ReplayMemory(self.memory_size)
        self.episode_durations = []

        
    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(self.episode_durations)
        plt.title("Training ...")
        plt.xlabel("Epsidoe")
        plt.ylabel("Duration")
        plt.plot(durations_t.numpy())

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.0001)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

            
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            # ReplayMemoryが小さすぎる場合は何もしない
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # non-final states
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))

        # 終了状態に対してbackpropをOFFにする？
        non_final_next_staes = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)

        # batches
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # compute Qt(s, a)
        Qsa_values = self.model(state_batch).gather(1, action_batch)

        # compute Vt+1(s)
        Vs_values = Variable(torch.zeros(self.batch_size).type(Tensor))
        Vs_values[non_final_mask] = self.model(non_final_next_staes).max(1)[0]
        Vs_values.volatile = False
        expected_Qsa_values = reward_batch + Vs_values * self.gamma

        # Loss
        loss = F.smooth_l1_loss(Qsa_values, expected_Qsa_values)

        # optimize
        self.optim.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()
            

    def act(self, state):
        sample = random.random()
        eps_t = self.eps_e + (self.eps_s - self.eps_e) * math.exp(-1.0 * self.steps_done / self.eps_d)
        self.steps_done += 1
        if sample > eps_t:
            return self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1]
        return LongTensor([[random.randrange(2)]])
        

    
if __name__ == "__main__":
    # unwrapped; to remove time limit
    env = gym.make("CartPole-v0").unwrapped
    plt.ion()

    # DQN
    agent = DQN_agent()

    num_episodes = 1000
    for i_ep in range(num_episodes):
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        for t in count():
            action = agent.act(state)
            # print(t, action[0, 0])

            _, reward, done, _ = env.step(action[0, 0])
            reward = Tensor([reward])

            # observe a new state
            last_screen = current_screen
            current_screen = get_screen(env)

            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # save into DQN memory
            agent.memory.push(state, action, next_state, reward)
            state = next_state

            # update
            agent.optimize_model()
            if done:
                agent.episode_durations.append(t + 1)
                agent.plot_durations()
                break

    # end
    print("Complete")
    env.render(close=True)
    env.close()
    plt.ioff()
    plt.show()
    plt.close()

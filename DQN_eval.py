import torch
import torch.nn as nn
import torch.autograd as autograd
from collections import namedtuple
import torch.nn.functional as F
import random
import torch.optim as optim
import gym
from gym import wrappers
import csv
import math
import argparse

#input parser
parser = argparse.ArgumentParser(description='DQN Carpole Example')
parser.add_argument('--model', type=str, default='Models/DQN_best.tar',
                    help='path to model to evaluate')
args = parser.parse_args()

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
REPLAY_SIZE = 5000
BATCH_SIZE = 128
TARGETQ_UPDATE = 50
num_episodes = 100
STEP = 250 #Max : 200
steps_done = 0

def Variable(data, *args, **kwargs):
    if use_cuda:
        data = data.cuda()
    return autograd.Variable(data,*args, **kwargs)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','done'))
#experience replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self,env,replayMemory):
        super(DQN,self).__init__()
        self.memory = replayMemory
        self.epsilon = EPS_START
        self.action_dim = env.action_space.n
        self.state_dim =  env.observation_space.shape[0]
        self.model = nn.Sequential(nn.Linear(self.state_dim,32),
                                nn.ReLU(),
                                nn.Linear(32,self.action_dim))
        self.targetModel = nn.Sequential(nn.Linear(self.state_dim,32),
                                nn.ReLU(),
                                nn.Linear(32,self.action_dim))
        self.updateTargetModel()
        self.targetModel.eval()

    def forward(self,x):
        return self.model(x)
    
    def target_forward(self,x):
        return self.targetModel(x)

    def updateTargetModel(self):
        #Assign weight to the target model
        self.targetModel.load_state_dict(self.model.state_dict())

    #epsilon greedy policy to select action
    def egreedy_action(self,state):
        global steps_done
        if self.epsilon >= EPS_END:
            self.epsilon *= EPS_DECAY
        steps_done += 1
        if random.random() > self.epsilon:
            return self.action(state)
        else:
            return LongTensor([[random.randrange(self.action_dim)]])
        
    def action(self,state):
        return self.forward(Variable(state)).detach().data.max(1)[1].view(1, 1)

    def loss(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        minibatch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(minibatch.state))
        action_batch = Variable(torch.cat(minibatch.action))
        reward_batch = Variable(torch.cat(minibatch.reward))
        done_batch = Variable(torch.cat(minibatch.done))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.forward(state_batch).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor),volatile=True)
        non_final_next_states = Variable(torch.cat([s for t,s in enumerate(minibatch.next_state) if done_batch[t]==0]))
        next_state_values[done_batch == 0] = self.forward(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # Undo volatility (which was used to prevent unnecessary gradients)
        expected_state_action_values = Variable(expected_state_action_values.data,volatile=False)

        loss = nn.MSELoss()
        loss = loss(torch.squeeze(state_action_values), expected_state_action_values)
        return loss

    def push(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def saveModel(self, name):
        torch.save(self.model.state_dict(), name)
    
    def loadModel(self, name):
        self.model.load_state_dict(torch.load(name))
        self.updateTargetModel()

def main():
    env = gym.make('CartPole-v0')
    memory = ReplayMemory(REPLAY_SIZE)
    dqn = DQN(env,memory)
    dqn.loadModel(args.model)
    total_reward = 0

    if use_cuda:
        dqn.model.cuda()
        dqn.targetModel.cuda()
    
    for i in range(num_episodes):
        state = env.reset()
        state = torch.from_numpy(state.reshape((-1, 4))).float()
        episode_reward = 0
        for j in range(STEP):
            action = dqn.action(state)
            state,reward,done,_ = env.step(int(action[0,0].data[0]))
            state = torch.from_numpy(state.reshape((-1, 4))).float()
            episode_reward += reward
            total_reward += reward
            if done:
                print('Episode: {} Evaluation Reward: {}'.format(i + 1, episode_reward))
                break
    avg_reward = total_reward / num_episodes
    print('Average Reward: {}'.format(avg_reward))

if __name__ == '__main__':
    main()

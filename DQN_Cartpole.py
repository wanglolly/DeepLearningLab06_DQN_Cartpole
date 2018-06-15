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
num_episodes = 1000
STEP = 300
TEST = 10
steps_done = 0

recordFileName = './DQN_Reward.csv'
recordFile = open(recordFileName, 'w')
recordCursor = csv.writer(recordFile)

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
            self.epsilon *= EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
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

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          minibatch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in minibatch.next_state
                                                if s is not None]),volatile=True)

        state_batch = Variable(torch.cat(minibatch.state))
        action_batch = Variable(torch.cat(minibatch.action))
        reward_batch = Variable(torch.cat(minibatch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.forward(state_batch).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor),volatile=True)
        next_state_values[non_final_mask] = self.target_forward(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # Undo volatility (which was used to prevent unnecessary gradients)
        expected_state_action_values = Variable(expected_state_action_values.data,volatile=False)

        #loss = F.smooth_l1_loss(torch.squeeze(state_action_values), expected_state_action_values)
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
    if use_cuda:
        dqn.model.cuda()
        dqn.targetModel.cuda()
    optimizer = optim.RMSprop(dqn.model.parameters(),0.0005)
    total_steps = 0
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.from_numpy(state.reshape((-1,4))).float()
        total_reward = 0
        for t in range(STEP):
            action = dqn.egreedy_action(state)
            next_state,reward,done,_ = env.step(int(action[0,0].data[0]))
            next_state = torch.from_numpy(next_state.reshape((-1,4))).float()
            total_reward += reward
            reward = Tensor([reward])
            final = LongTensor([done])
            dqn.push(state,action,next_state,reward,final)
            state = next_state
            loss = dqn.loss()
            total_steps += 1
            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                #for param in dqn.model.parameters():
                #    param.grad.data.clamp_(-1, 1)
                optimizer.step()
            if total_steps % TARGETQ_UPDATE == 0:
                dqn.updateTargetModel()
            if done:
                if loss is not None:
                    print(str(episode) + "\tSTEP: " + str(t) + "\tLoss: " + str(float(loss.data[0].cpu())) + "\tReward: " + str(total_reward))
                break
        if loss is not None:
            header = [episode, total_reward, str(float(loss.data[0].cpu()))]
            recordCursor.writerow(header)

        #if(episode + 1) % TARGETQ_UPDATE == 0:
        #    dqn.updateTargetModel()

        if (episode + 1) % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                state = torch.from_numpy(state.reshape((-1, 4))).float()
                for j in range(STEP):
                    #env.render()
                    action = dqn.action(state)
                    state,reward,done,_ = env.step(int(action[0,0].data[0]))
                    state = torch.from_numpy(state.reshape((-1, 4))).float()
                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward / TEST
            print('Episode: {} Evaluation Average Reward: {}'.format(episode + 1, avg_reward))

    dqn.saveModel('Models/DQN_' + str(episode) + '.tar')
if __name__ == '__main__':
    main()

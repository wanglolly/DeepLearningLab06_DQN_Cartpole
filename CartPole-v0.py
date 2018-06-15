import torch
import torch.nn as nn
import torch.autograd as autograd
from collections import namedtuple
import torch.nn.functional as F
import random
import torch.optim as optim
import gym
from gym import wrappers
torch.set_default_tensor_type('torch.FloatTensor')

GAMMA = 0.9
EPS_START = 0.5
EPS_END = 0.01
REPLAY_SIZE = 10000
BATCH_SIZE = 64
USE_CUDA = True
num_episodes = 10000
STEP = 600
TEST = 10

def Variable(data, *args, **kwargs):
    if USE_CUDA:
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
        self.fc1 = nn.Linear(self.state_dim,20)
        self.fc2 = nn.Linear(20,self.action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    #epsilon greedy policy to select action
    def egreedy_action(self,state):
        if random.random() <= self.epsilon:
            return torch.LongTensor([[random.randrange(self.action_dim)]])
        else:
            return self.action(state)
        self.epsilon -= (EPS_START - EPS_END) / 10000

    def action(self,state):
        return self.forward(Variable(state)).detach().data.max(1)[1].cpu()

    def loss(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        minibatch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(minibatch.state))
        action_batch = Variable(torch.cat(minibatch.action))
        reward_batch = Variable(torch.cat(minibatch.reward))
        done_batch = Variable(torch.cat(minibatch.done))

        #Q(s,a,theta)
        state_action_values = self.forward(state_batch).gather(1, action_batch)
        next_state_values = Variable(torch.zeros(BATCH_SIZE))
        non_final_next_states = Variable(torch.cat([s for t,s in enumerate(minibatch.next_state) if done_batch[t]==0]))

        #max Q(s',a',theta),if s' is a terminal state,return 0
        next_state_values[done_batch == 0] = self.forward(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        return loss

    def push(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

def main():
    env = gym.make('CartPole-v0')
    memory = ReplayMemory(REPLAY_SIZE)
    dqn = DQN(env,memory)
    if USE_CUDA:
        dqn.cuda()
    optimizer = optim.Adam(dqn.parameters(),1e-4)
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.from_numpy(state.reshape((-1,4))).float()
        for t in range(STEP):
            action = dqn.egreedy_action(state)
            next_state,reward,done,_ = env.step(action[0,0])
            next_state = torch.from_numpy(next_state.reshape((-1,4))).float()
            reward = torch.Tensor([reward])
            final = torch.LongTensor([done])
            dqn.push(state,action,next_state,reward,final)
            state = next_state
            optimizer.zero_grad()
            loss = dqn.loss()
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            if done:
                break

        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                state = torch.from_numpy(state.reshape((-1, 4))).float()
                for j in range(STEP):
                    env.render()
                    action = dqn.action(state)
                    state,reward,done,_ = env.step(action[0,0])
                    state = torch.from_numpy(state.reshape((-1, 4))).float()
                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward / TEST
            print 'episode: ' ,episode,' Evaluation Average Reward: ',avg_reward
            if avg_reward >= 200.0:
                break

    env = wrappers.Monitor(env,'CartPole-v0-experiment-1')
    for i in xrange(100):
        state = env.reset()
        state = torch.from_numpy(state.reshape((-1, 4))).float()
        for j in range(STEP):
            env.render()
            action = dqn.action(state)
            state, reward, done, _ = env.step(action[0, 0])
            state = torch.from_numpy(state.reshape((-1, 4))).float()
            total_reward += reward
            if done:
                break
    env.close()

if __name__ == '__main__':
    main()

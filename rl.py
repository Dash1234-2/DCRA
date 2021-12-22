import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import torch.optim as optim
from collections import deque

MAX_EPISODES = 1000
MAX_EP_STEPS = 10
LR_A = 0.0005   
LR_C = 0.0005  
MEMORY_CAPACITY =10000 
BATCH_SIZE = 32 
update_iteration=200
GAMMA = 0.9     
TAU = 0.005     

directory='./DDPG/model/'
class Actor(nn.Module):
    def __init__(self,s_dim,a_dim):
        super(Actor,self).__init__()
        self.l1=nn.Linear(s_dim,30)
        self.l2=nn.Linear(30,a_dim)
        self.l2.weight.data.normal_(0,0.1)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=torch.tanh(self.l2(x))
        return x


class Critic(nn.Module):
    def __init__(self,s_dim,a_dim):
        super(Critic,self).__init__()
        n_l1=30
        self.l1=nn.Linear(s_dim+a_dim,n_l1)
        self.l2=nn.Linear(n_l1,1)
        self.l2.weight.data.normal_(0,0.1)
    def forward(self,x,ba):      
        x=F.relu(self.l1(torch.cat([x,ba],1)))    
        x=self.l2(x)                 
        return x


class DDPG(object):
    def __init__(self,a_dim,s_dim):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer=0
        self.s_dim,self.a_dim=s_dim,a_dim
 
        self.actor=Actor(s_dim,a_dim)
        self.actor_target=Actor(s_dim,a_dim)

        self.critic=Critic(s_dim,a_dim)
        self.critic_target=Critic(s_dim,a_dim)

        self.ctrain = torch.optim.Adam(self.critic.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.actor.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()

    def sample(self):
        indics = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indics, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])
        a = self.actor(bs)
        q = self.critic(bs,a)

        loss_a = -torch.mean(q) 
        
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.actor_target(bs_)  
        q_ = self.critic_target(bs_,a_)  
        q_target = br+GAMMA*q_  
        q_v = self.critic(bs,ba)
        td_error = self.loss_td(q_target,q_v)
       
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic.' + x + '.data)')

    def push(self, s,a,r,s_):
        data=np.hstack((s,a,[r],s_))
        index=self.pointer%MEMORY_CAPACITY
        self.memory[index,:]=data
        self.pointer+=1

    def save(self):
            torch.save(self.actor.state_dict(), directory + 'actor.pth')
            torch.save(self.critic.state_dict(), directory + 'critic.pth')

    def load(self):
            self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
            self.critic.load_state_dict(torch.load(directory + 'critic.pth'))

    def choose_action(self,s,var):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        a=self.actor(s)[0].detach()  
        a_ = np.clip(np.random.normal(a, var), 0.05, 1)  
        return a_
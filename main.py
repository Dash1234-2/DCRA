#coding=utf-8
from rl import DDPG
from env import Trans
import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from lib.logger import get_logger
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
V =6 
NUM_VEHICLES =20 
NUM_MEC = 5 # RSU+BS
MAX_EP_STEPS = V * (NUM_MEC-1) 
MEMORY_CAPACITY = 2000 
MAX_EPISODES = 1000

env = Trans(NUM_VEHICLES,NUM_MEC)
s_dim = env.states.shape[0]
a_dim = env.action.shape[0]
print(env.states.shape,env.action.shape)
print(env.states.shape[0],env.action.shape[0])

rl=DDPG(a_dim,s_dim)
logger=get_logger('./log',debug=False)

def train():
    var=3
    plot_re=[]
    for i in range(MAX_EPISODES):
        s=env.reset()
        ep_reward=0
        for j in range(MAX_EP_STEPS):
            K=int(j/V)
            a=rl.choose_action(s,var) 
            s_,r=env.step(s,a,K)   
            
            rl.push(s, a, r, s_)
    
            if rl.pointer>=MEMORY_CAPACITY:
                var*=.9998  
                rl.sample()
            s=s_ 
            ep_reward+=r
            if j == MAX_EP_STEPS - 1:
               
                logger.info('Episode:{0},  Reward:{1} ,Explore:{2}'.format(i+1,float(ep_reward),float(var)))
                plot_re.append(ep_reward)
              
                break   
    return plot_re


def eval():
    rl.load()  
    while True:
        s = env.reset()
        for _ in range(200):
            a = rl.choose_action(s)
            s,r = env.step(s,a)

if __name__=='__main__':
    ON_TRAIN=True
    plot_re=[]
    t1=time.time()
    if ON_TRAIN:
        plot_re=train()
    else:
        eval()
    with open('result/result_DCRA.csv', 'w',newline='') as csvfile:
        result = csv.writer(csvfile, dialect='excel')
        result.writerow(['num_vehicles','episodes','reward','Speed'])
        for s in range(len(plot_re)):
            result.writerow([20,s+1,plot_re[s],30])
    plt.plot(np.arange(len(plot_re)), plot_re)
    plt.ylabel('Reward')
    plt.xlabel('Episodes')
    plt.savefig("DCRA.png")
    plt.show()
    logger.info('Running time:{}'.format(time.time() - t1))
    
    




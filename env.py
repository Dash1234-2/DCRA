import numpy as np
import torch
import random
from torch.autograd import Variable


class Trans(object):
    state_map = None  # state pool
    def __init__(self,user_num,MEC_num):
        self.user_num = user_num
        self.MEC_num = MEC_num
        self.Feature_len = 6 

        self.MEC_Bandwidth = 10 # MHz
        self.BS_Bandwidth = 20 # MHz
        self.MEC_CPU = 5 # 5GHz
        self.BS_CPU = 10 # 10GHz

        self.MEC_Cache = 5 # 5GB
        self.BS_Cache = 10 # 10GB

        self.MEC_Comp_Cost = 0.3 # units/J
        self.BS_Comp_Cost = 1

        self.COMP_Size = [0.15,0.25,0.3,0.4,0.45,0.6] 
        self.Content_Size = [0.15,0.25,0.3,0.4,0.45,0.6] 

        self.CONTENT_Pop = [1,2,3,4,5,6] 
        self.COMP_Resource = [0.5,0.6,0.7,0.8,1.2] 

        self.comp_energy=9*1e-5      
        self.cache_energy=2*1e-5  

        self.action=Variable(torch.zeros([MEC_num * 4 * user_num, 1], dtype=torch.uint8,requires_grad=False))
        self.states = Variable(torch.zeros([self.Feature_len * user_num + 3 * MEC_num, 1], dtype=torch.float32))

    def get_reward(self,action,obervation,K):
            for i in range(0, self.MEC_num * 4 * self.user_num, self.MEC_num * 4):
                BS_comp = action[i]
                MEC_comp = action[i+4*(K+1)]
                BS_band_comp = action[i+2]
                MEC_band_comp = action[i+4*(K+1)+2]
                v_id = int(i/(self.MEC_num * 4))*6

                v_comp_re = obervation[v_id+0]
                v_comp_size = obervation[v_id+1]
                BS_comp_rate = float((BS_comp * v_comp_size) /v_comp_re)
                MEC_comp_rate = float((MEC_comp * v_comp_size) /v_comp_re)
               
                Bs_comp_energy=BS_comp_rate*self.comp_energy*3.6*10**3*1024

                RSU_comp_energy=MEC_comp_rate*self.comp_energy*3.6*10**3*1024

                comp_cost=Bs_comp_energy*self.BS_Comp_Cost+RSU_comp_energy*self.MEC_Comp_Cost
               
                BS_cache = action[i+1]
                MEC_cache = action[i + 4 * (K + 1)+1]

                BS_cache_energy=BS_cache*self.cache_energy*3.6*10**3*1024

                RSU_cache_energy=MEC_cache*self.cache_energy*3.6*10**3*1024

                cache_energy=BS_cache_energy+RSU_cache_energy

                cache_cost=BS_cache_energy*self.BS_Comp_Cost+RSU_cache_energy*self.MEC_Comp_Cost
                tmp_pop = int(self.Feature_len*i/(self.MEC_num * 3) + 3)
                Com_pop = obervation[v_id+3]
                reward = -(cache_cost+comp_cost)
            return reward
    
    def reset(self):
        ob = []
        for j in range(self.user_num):
            Comp_re = self.COMP_Resource[1]
            Comp_Size = self.COMP_Size[1]
            Content_Size = self.COMP_Size[1]
            Content_Pop = self.CONTENT_Pop[1]
            tmp = [Comp_re, Comp_Size, Content_Size, Content_Pop, Comp_Size, Content_Size]
            ob = ob + tmp
        BS_State = [self.BS_CPU,self.BS_Cache,self.BS_Bandwidth]
        ob = ob + BS_State
        for i in range(self.MEC_num-1):
            ob = ob + [self.MEC_CPU, self.MEC_Cache, self.MEC_Bandwidth]
        #print('reset',ob)
        return np.array(ob)

    def step(self,observation,action,K):
        action = list(action)  
        observation_ = list(observation)
        reward = self.get_reward(action,observation_,K)
        for i in range(0, self.MEC_num * 4 * self.user_num, self.MEC_num * 4):
            BS_comp = action[i]
            MEC_comp = action[i+4*(K+1)]
            v_id = int(i/(self.MEC_num * 4))*6
            v_comp_re = observation_[v_id+0]
            v_comp_size = observation_[v_id+1]
            BS_comp_rate = float((BS_comp * v_comp_size) /v_comp_re)
            MEC_comp_rate = float((MEC_comp * v_comp_size) /v_comp_re)
            Computing_Task_Size = float((BS_comp_rate+ MEC_comp_rate))
            observation_[v_id + 4] = observation_[v_id + 4] - Computing_Task_Size

            BS_cache = action[i+1]
            MEC_cache = action[i + 4 * (K + 1)+1]
            Content_Task_Size = BS_cache + MEC_cache
            observation_[v_id + 5] = observation_[v_id + 5] - Content_Task_Size
            BS_band_comp = action[i+2]
            MEC_band_comp = action[i+4*(K+1)+2]
            BS_band_comm = action[i + 3]
            MEC_band_comm = action[i + 4 * (K + 1)+3]
            BS_band_total = BS_band_comp + BS_band_comm
            MEC_band_total = MEC_band_comp + MEC_band_comm

            observation_[self.user_num * self.Feature_len] = observation_[self.user_num * self.Feature_len] - BS_comp
            observation_[self.user_num * self.Feature_len+1] = observation_[self.user_num * self.Feature_len+1] - BS_cache
            observation_[self.user_num * self.Feature_len+2] = observation_[self.user_num * self.Feature_len+2] - BS_band_total
            observation_[self.user_num * self.Feature_len + (K+1)*3] = observation_[self.user_num * self.Feature_len+ (K+1)*3] - MEC_comp
            observation_[self.user_num * self.Feature_len + (K+1)*3+1] = observation_[self.user_num * self.Feature_len+ (K+1)*3+1] - MEC_cache
            observation_[self.user_num * self.Feature_len + (K+1)*3+2] = observation_[self.user_num * self.Feature_len+ (K+1)*3+2] - MEC_band_total
        return np.array(observation_), reward
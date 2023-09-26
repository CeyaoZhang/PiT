import numpy as np

import pickle
import os
import torch
import random
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
from collections import namedtuple, deque
#from optim_PhC import ReplayMemory

# declare transition and experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
offset={0:np.array([25,0,0,0,0,0,0,0],dtype=float),1:np.array([-25,0,0,0,0,0,0,0],dtype=float),\
    2:np.array([0,2.5,0,0,0,0,0,0],dtype=float),3:np.array([0,-2.5,0,0,0,0,0,0],dtype=float),4:np.array([0,0,2.5,0,0,0,0,0],dtype=float),\
    5:np.array([0,0,-2.5,0,0,0,0,0],dtype=float),6:np.array([0,0,0,2.5,0,0,0,0],dtype=float),7:np.array([0,0,0,-2.5,0,0,0,0],dtype=float),\
    8:np.array([0,0,0,0,0.005,0,0,0],dtype=float),9:np.array([0,0,0,0,-0.005,0,0,0],dtype=float),10:np.array([0,0,0,0,0,0.005,0,0],dtype=float),\
    11:np.array([0,0,0,0,0,-0.005,0,0],dtype=float),12:np.array([0,0,0,0,0,0,0.005,0],dtype=float),13:np.array([0,0,0,0,0,0,-0.005,0],dtype=float),\
    14:np.array([0,0,0,0,0,0,0,2.5],dtype=float),15:np.array([0,0,0,0,0,0,0,-2.5],dtype=float)}
class ReplayMemory(object):
    """declare the replay buffer"""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if __name__=='__main__':
    file_path = os.getcwd()+'/data_samples/data_samples/PCSEL_dataset/'
    transition=dict()
    paths=[]
    max_ep=0
    for file in os.listdir(file_path):
        flag=False#flag of terminal
        if file.endswith('.npy'):
            a = np.load(file_path+file, allow_pickle=True).item()
            # transition[file]=len(a)
            i=0
            x=[]
            y=[]
            z=[]
            w=[]
            v=[]
            z=[]
            # first_obs=None
            for item in a.memory:
                i+=1
                x.append(item[0].detach().cpu().numpy())#o
                #y.append(item[1].detach().cpu().numpy()[0][0])#a number
                one_hot_matrix = np.eye(16)
                integer = item[1].detach().cpu().numpy()[0][0]
                y.append(one_hot_matrix[integer])
                temp=item[2].detach().cpu().numpy() if item[2] is not None else None
                w.append(item[3].detach().cpu().numpy()[0])#r
                if temp is None:
                    v.append(True)
                    #z.append(x[-1]+offset[y[-1]]) #number
                    z.append(x[-1]+np.argmax(y[-1]))
                    trans={'observations':np.array(x),'actions':np.array(y),'rewards':np.array(w),'next_observation':np.array(z)\
                    ,'terminals':np.array(v)}
                    paths.append(trans)
                    print(len(trans['rewards']))
                    # current_first=trans['observations'][0]
                    # print(first_obs)
                    # if first_obs is None:
                    #     first_obs=current_first
                    # elif not np.array_equal(current_first,first_obs):
                    #     raise ValueError
                    if len(trans['rewards'])>max_ep:
                        max_ep=len(trans['rewards'])
                else:
                    v.append(False)
                    z.append(temp)#o',
                # one_hot = np.eye(16)[y[0]]
    print("number",len(paths))
    # with open(f'DTstat.pkl', 'wb') as f:
    #     pickle.dump(transition, f)
    # with open(f'DTdict.pkl','wb') as f:
    #     pickle.dump(transition,f)
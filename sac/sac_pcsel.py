from spinningup.spinup.algos.pytorch.sac import sac 
from spinningup.spinup.algos.pytorch.sac import core 
from gym import register
from torch.utils.tensorboard import SummaryWriter
import torch
import gym
from functools import partial
import os 
register(
    id='Fdtd_NB-v1',
    entry_point='envs:FdtdEnv_v1',
    max_episode_steps=250,
    reward_threshold=250.0,
)

writer = SummaryWriter()  # log the training process

# if GPU is to be used
#print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
"""
save_add='/home/ondemand/220019012/RLcode/sacres/sac2/'
assert not os.path.exists(save_add)
os.makedirs(save_add)
Agent=sac.sac(lambda : gym.make('Fdtd_NB-v1'),save_address=save_add,start_steps=500,max_ep_len=250)
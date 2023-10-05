from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
import torch
from envs.fdtd_env import FdtdEnv
import numpy as np
import pickle

datapath='./test_onehot.pkl'
env=FdtdEnv()
state_dim = env.observation_space.shape[0]
model_path='./output/20230928-134025.pth'
act_dim=16

max_ep_len=250
scale=1
target_rew=92
mode='normal'
with open(datapath, 'rb') as f:
    trajectories = pickle.load(f)

device='cuda'
states=[]
for path in trajectories:
    states.append(path['observations'])
states = np.concatenate(states, axis=0)
state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
settings={
            "state_dim":state_dim,
            "act_dim":act_dim,
            "max_length":20,
            "max_ep_len":max_ep_len,
            "hidden_size":128,
            "n_layer":3,
            "n_head":1,
            "n_inner":4*128,
            "activation_function":'relu',
            "n_positions":1024,
            "resid_pdrop":0.1,
            "attn_pdrop":0.1
            }
model=DecisionTransformer.load(settings,model_path)
with torch.no_grad():
    rtg, length,states1,rewards = evaluate_episode_rtg(
            env,
            state_dim,
            act_dim,
            model,
            max_ep_len=max_ep_len,
            scale=scale,
            target_return=target_rew/scale,
            mode=mode,
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            )
    mydict={
            "rtg":rtg,
            "length":length,
            "states":states1,
            "scores":rewards
        }
    path=f'./eval_target={target_rew}.pkl'
    with open(path,'wb') as f:
        pickle.dump(mydict,f)
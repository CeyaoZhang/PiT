"""
custom gym environment developed for invoking FDTD simulations used in DQN.
Author: Renjie Li. March 2023 @ NOEL.
"""

import FdtdRlNanobeam
import random
import gym
from gym import spaces, logger
from gym import utils
from gym.utils import seeding
from collections import namedtuple, deque
from itertools import count
import subprocess, time, signal
import numpy as np
import sys
import os

# sys.path.append("D:\\Program Files\\Lumerical\\FDTD\\api\\python\\")  # Default windows lumapi path
# sys.path.append(os.path.dirname(__file__))  # Current directory


class FdtdEnv(gym.Env):
    """
    Makes changes to the physical parameters of PCSEL to optimize optical responses.
    Invokes an FDTD session to take in (dx, dy, dr) and compute the resulting Q factor.

    Observations:
    Type: Box(3)
    Num     Observation                  Min                     Max
    0       net x change                -20 nm                   20 nm
    1       net y change                -20 nm                   20 nm
    2       net r change                -10 nm                   10 nm

    Actions:
    Type: Discrete(6)
    Num   Action
    0     increase x by 0.5 nm
    1     decrease x by 0.5 nm
    2     increase y by 0.5 nm
    3     decrease y by 0.5 nm
    4     increase r by 0.25 nm
    5     decrease r by 0.25 nm

    reward:
    r = 200 - (Q_target - Q)*E-7
    where Q_target = 1E+9 is the optimal Q to be achieved.

    reset:
    At the end of each episode, states are returned to zeros.

    Episode termination:
    Episode length is more than 300,
    net x change is over +- 20nm,
    net y change is over +- 20nm,
    net r change is over +- 10nm.
    Solved requirement:
    considered solved when the reward >= 150 (i.e., Q >= 0.5E9).
    """

        # len = 2000E-9  #width
        # t = 450E-9
        # t_1 = 100E-9
        # t_2 = 0
        # t_3 = 315E-9
        # t_4 = 5000E-9
        # n_1 = 3.2035
        # n_2 = 3.4038 #GaAs (from https://refractiveindex.info)
        # n_3 = 3.415
        # n_4 = 3.2035
        # leng = 0.52  #radius
        # leng2 = 0.406 #for triangular holes
        # a = 400E-9


    metadata = {'render.modes': ['human']}

    def __init__(self):
        # limits for net geometrical changes (states). Less important variables are commented out.
        self.maxDeltaLen = 1000  # 2000E-9  #width
        self.maxDeltaT = 100    # 450E-9
        self.maxDeltaT1 = 50   # 100E-9
        #self.maxDeltaT2 = 0
        self.maxDeltaT3 = 100   # 315E-9
        #self.maxDeltaT4 = 5000E-9   #cladding layer, no need to change
        self.maxDeltaN1 = 0.15  # 3.2035
        #self.maxDeltaN2 = 3.4038 #GaAs, don't need to change this item
        self.maxDeltaN3 = 0.15  # 3.415  active layer
        #self.maxDeltaN4 = 3.2035  #this is equal to self.maxDeltaN1
        self.maxDeltaLeng = 0.3    #0.52   #radius
        #leng2 = 0.406 #for triangular holes
        self.maxDeltaA = 100   # 400E-9
        
        # actions to take (i.e. alter the geometrical parameters)
        self.deltaLen = 25
        self.deltaTA = 2.5
        self.deltaN = 0.005 #for n and leng

        high = np.array(
            [
                self.maxDeltaLen * 1.5,
                self.maxDeltaT * 1.5,
                self.maxDeltaT1 * 1.5,
                self.maxDeltaT3 * 1.5,
                self.maxDeltaN1 * 1.5,
                self.maxDeltaN3 * 1.5,
                self.maxDeltaLeng * 1.5,
                self.maxDeltaA * 1.5
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # best geometrical shift values found so far (taken to be initial values)
        self.len = 125.  # 2000E-9  #width
        self.t = 0.   # 450E-9
        self.t1 = -50.   # 100E-9
        self.t3 = -27.5   # 315E-9
        self.n1 = 0.  # 3.2035
        self.n3 = 0.  # 3.415 
        self.leng = -0.005   #0.52   #radius
        self.a = 0.   # 400E-9
        
        # optimization goal
        self.Q_goal = 5.0e+6  
        self.area_goal = 3.6e-13  #area >= 3.6e-13 m^2
        self.lam_goal = 1310.0   #or 980 nm
        self.P_goal = 0.3    #output power/injecting power >= 30% 
        self.div_goal = 1.0    # divergence angle <= 1 degree
        
        #other setup
        #self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        netDLen, netDT, netDT1, netDT3, netDN1, netDN3, netDLeng, netDA = self.state

        if action == 0:
            netDLen = netDLen + self.deltaLen

        elif action == 1:
            netDLen = netDLen - self.deltaLen

        elif action == 2:
            netDT = netDT + self.deltaTA

        elif action == 3:
            netDT = netDT - self.deltaTA

        elif action == 4:
            netDT1 = netDT1 + self.deltaTA

        elif action == 5:
            netDT1 = netDT1 - self.deltaTA

        elif action == 6:
            netDT3 = netDT3 + self.deltaTA

        elif action == 7:
            netDT3 = netDT3 - self.deltaTA

        elif action == 8:
            netDN1 = netDN1 + self.deltaN

        elif action == 9:
            netDN1 = netDN1 - self.deltaN

        elif action == 10:
            netDN3 = netDN3 + self.deltaN

        elif action == 11:
            netDN3 = netDN3 - self.deltaN

        elif action == 12:
            netDLeng = netDLeng + self.deltaN

        elif action == 13:
            netDLeng = netDLeng - self.deltaN
        
        elif action == 14:
            netDA = netDA + self.deltaTA

        elif action == 15:
            netDA = netDA - self.deltaTA

        # perform an action in fdtd and compute Q factor
        FR = FdtdRlNanobeam()
        c = 1e-9  # define conversion from m to nm
        Q, lam, power, area, div_angle = FR.adjustdesignparams(netDLen*c, netDT*c, netDT1*c, netDT3*c, netDN1, netDN3, netDLeng, netDA*c)

        # update the state
        self.state = (netDLen, netDT, netDT1, netDT3, netDN1, netDN3, netDLeng, netDA)

        done = bool(
            netDLen < -self.maxDeltaLen
            or netDLen > self.maxDeltaLen
            or netDT < -self.maxDeltaT
            or netDT > self.maxDeltaT
            or netDT1 < -self.maxDeltaT1
            or netDT1 > self.maxDeltaT1
            or netDT3 < -self.maxDeltaT3
            or netDT3 > self.maxDeltaT3
            or netDN1 < -self.maxDeltaN1
            or netDN1 > self.maxDeltaN1
            or netDN3 < -self.maxDeltaN3
            or netDN3 > self.maxDeltaN3
            or netDLeng < -self.maxDeltaLeng
            or netDLeng > self.maxDeltaLeng
            or netDA < -self.maxDeltaA
            or netDA > self.maxDeltaA
        )

        gamma = 1
        eps = 1 
        beta = 100
        alpha = 100
        eta = 20
        # calculate the score
        if not done:
            r1 = gamma * (1 - (self.Q_goal - Q) / self.Q_goal)
            r2 = eps / (1 - abs(self.lam_goal - lam) / self.lam_goal)
            r3 = beta * (1 - (self.area_goal - area) / self.area_goal)
            r4 = alpha * (1 - (self.P_goal- power) / self.P_goal)
            r5 = eta * (1 + (self.div_goal - div_angle) / self.div_goal)
            r_total = r1 + r2 + r3 + r4 + r5
            score = np.float32(r_total)
        elif self.steps_beyond_done is None:
            # net changes out of limit, game over
            self.steps_beyond_done = 0
            r1 = gamma * (1 - (self.Q_goal - Q) / self.Q_goal)
            r2 = eps / (1 - abs(self.lam_goal - lam) / self.lam_goal)
            r3 = beta * (1 - (self.area_goal - area) / self.area_goal)
            r4 = alpha * (1 - (self.P_goal- power) / self.P_goal)
            r5 = eta * (1 + (self.div_goal - div_angle) / self.div_goal)
            r_total = r1 + r2 + r3 + r4 + r5
            score = np.float32(r_total)
            print('State out of range, done! Restarting a new episode...')

        # if not done:
        #     r1 = gamma * (50 - (self.Q_goal - Q) * 1e-5)
        #     r2 = eps / abs(self.lam_goal-lam)
        #     r3 = beta * (36 -  (self.area_goal - area) * 1e14)
        #     r4 = alpha * (power - self.P_goal)
        #     r5 = eta * (self.div_goal - div_angle)
        #     r_total = r1 + r2 + r3 + r4 + r5
        #     score = np.float32(r_total)
        # elif self.steps_beyond_done is None:
        #     # net changes out of limit, game over
        #     self.steps_beyond_done = 0
        #     r1 = gamma * (50 - (self.Q_goal - Q) * 1e-5)  #1-50
        #     r2 = eps / abs(self.lam_goal-lam)   #decimal or 1-10
        #     r3 = beta * (36 -  (self.area_goal - area) * 1e14)  #1-36
        #     r4 = alpha * (power - self.P_goal)   #decimal
        #     r5 = eta * (self.div_goal - div_angle)  #decimal
        #     r_total = r1 + r2 + r3 + r4 + r5
        #     score = np.float32(r_total)
        #     print('State out of range, done! Restarting a new episode...')

        print('\nQ factor: {:.3f}, resonance lambda: {:.2f}, power: {:.4f}, area: {:.4e}, divergence: {:.4f}\n'.format(Q, lam, power,area, div_angle))

        print('score: {:.5f}, state: {}\n'.format(score, self.state))

        return np.array(self.state, dtype=np.float32), score, done, {}

    def reset(self):
        # self.state = np.zeros((4,), dtype=np.float32)
        self.state = (self.len, self.t, self.t1, self.t3, self.n1, self.n3, self.leng, self.a)
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)









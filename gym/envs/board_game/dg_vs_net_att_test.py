import numpy as np
import gym
from gym import spaces
from baselines import deepq

NODE_COUNT = 30
DEF_INPUT_SIZE = 240
ATT_INPUT_SIZE = 241

ATT_NETWORK = None
ATT_SESS = None
CUR_ATT_STRAT = "dg_dqmlp_rand30NoAnd_B_att_fixed.pkl"

class VsNetworkTest(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(self):
        self.action_space = spaces.Discrete(NODE_COUNT + 1)
        my_shape = (DEF_INPUT_SIZE,)
        self.observation_space = \
            spaces.Box(np.zeros(my_shape), np.ones(my_shape))

    def _reset(self):
        global ATT_NETWORK
        global ATT_SESS
        ATT_NETWORK, _, ATT_SESS = deepq.load_for_multiple_nets(CUR_ATT_STRAT)

        def_obs = np.zeros((1, DEF_INPUT_SIZE))
        return def_obs

    def _step(self, action):
        att_obs = np.zeros((1, ATT_INPUT_SIZE))
        with ATT_SESS.as_default():
            att_act = ATT_NETWORK(att_obs)
            print(att_act)

        def_obs = np.zeros((1, DEF_INPUT_SIZE))
        state_dict = {'state': def_obs[:]}
        is_done = True
        return def_obs, is_done, state_dict

    def _render(self, mode='human', close=False):
        if close:
            return
        print("")

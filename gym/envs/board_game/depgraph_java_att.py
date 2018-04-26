'''
Dependency graph game, from attacker view.

Requirements:
    Py4J        https://www.py4j.org/download.html
    OpenAI Gym  https://github.com/openai/gym#installation
'''

from py4j.java_gateway import JavaGateway
import numpy as np
import gym
from gym import spaces

NODE_COUNT = 30
AND_NODE_COUNT = 5
EDGE_TO_OR_NODE_COUNT = 100
OBS_LENGTH = 1
JAVA_GAME = None
GATEWAY = None

class DepgraphJavaEnvAtt(gym.Env):
    """
    Depgraph game environment. Play against a fixed opponent.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        # https://www.py4j.org/getting_started.html
        global GATEWAY
        GATEWAY = JavaGateway()
        global JAVA_GAME
        JAVA_GAME = GATEWAY.entry_point.getGame()

        # One action for each AND node or, edge to OR node, and "pass".
        # action space is {0, . . ., (AND_NODE_COUNT + EDGE_TO_OR_NODE_COUNT)}.
        action_count = AND_NODE_COUNT + EDGE_TO_OR_NODE_COUNT + 1
        self.action_space = spaces.Discrete(action_count)

        observation = self.reset()
        # convert from JavaMember object to JavaList
        observation = observation[:]
        my_shape = (len(observation),)
        self.observation_space = \
            spaces.Box(np.zeros(my_shape), np.ones(my_shape))

    def _reset(self):
        result_values = JAVA_GAME.reset()
        # result_values is a Py4J JavaList -> should convert to Python list
        return np.array([x for x in result_values])

    def _step(self, action):
        # action is a numpy.int64, need to convert to Python int before using with Py4J
        action_scalar = np.asscalar(action)
        # {1, . . ., AND_NODE_COUNT} correspond to AND nodes,
        # {AND_NODE_COUNT + 1, . . ., AND_NODE_COUNT + EDGE_TO_OR_NODE_COUNT}
        # correspond to edges to OR nodes,
        # and {AND_NODE_COUNT + EDGE_TO_OR_NODE_COUNT + 1} corresponds to "pass".
        action_id = action_scalar + 1
        return DepgraphJavaEnvAtt.step_result_from_flat_list(JAVA_GAME.step(action_id))

    @staticmethod
    def step_result_from_flat_list(a_list):
        '''
        Convert a flat list input, a_list, to the observation, reward,
        is_done, and state dictionary.
        a_list will be a list of floats, of length (game_size + 2).

        The first game_size elements of a_list represent the game state.

        The next element represents the reward, in R.

        The last element represents whether the game is done, in {0.0, 1.0}.
        '''
        game_size = \
            (AND_NODE_COUNT + EDGE_TO_OR_NODE_COUNT) * 2 + NODE_COUNT * OBS_LENGTH + 1

        obs_values = a_list[:game_size]
        # obs_values is a Py4J JavaList -> should convert to Python list
        obs = np.array([x for x in obs_values])

        reward = a_list[game_size]

        tolerance = 0.01
        is_done = abs(a_list[game_size + 1] - 1) < tolerance

        state_dict = {'state': obs[:]}
        return obs, reward, is_done, state_dict

    def _render(self, mode='human', close=False):
        if close:
            return
        print(JAVA_GAME.render())

    def get_opponent_reward(self):
        '''
        Get the total discounted reward of the opponent (defender) in the current game.
        '''
        return JAVA_GAME.getOpponentTotalPayoff()

    def get_self_reward(self):
        '''
        Get the total discounted reward of self (attacker) in the current game.
        '''
        return JAVA_GAME.getSelfTotalPayoff()

    def close_gateway(self):
        GATEWAY.close()
        GATEWAY.close_callback_server()
        GATEWAY.shutdown()

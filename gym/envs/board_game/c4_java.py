'''
Game of Connect Four.

Requirements:
    Py4J        https://www.py4j.org/download.html
    OpenAI Gym  https://github.com/openai/gym#installation
'''

from py4j.java_gateway import JavaGateway
import numpy as np
import gym
from gym import spaces

HEIGHT = 6
WIDTH = 7
JAVA_BOARD = None
GATEWAY = None

class C4JavaEnv(gym.Env):
    """
    Connect Four environment. Play against a fixed opponent.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        # https://www.py4j.org/getting_started.html
        global GATEWAY
        GATEWAY = JavaGateway()
        global JAVA_BOARD
        JAVA_BOARD = GATEWAY.entry_point.getBoard()

        # One action for each board position or resign
        # action space is {0, . . ., WIDTH}, indicating column or resign
        self.action_space = spaces.Discrete(WIDTH)

        observation = self.reset()
        # convert from JavaMember object to JavaList
        observation = observation[:]
        my_shape = (len(observation),)
        self.observation_space = \
            spaces.Box(np.zeros(my_shape), np.ones(my_shape))

    def _reset(self):
        result_values = JAVA_BOARD.reset()
        # result_values is a Py4J JavaList -> should convert to Python list
        return [x for x in result_values]

    def _step(self, action):
        # https://www.py4j.org/advanced_topics.html#collections-conversion
        action_for_java = GATEWAY.jvm.java.util.ArrayList()
        # action is a numpy.int64, need to convert to Python int before using with Py4J
        action_scalar = np.asscalar(action)
        action_for_java.add(action_scalar)
        return C4JavaEnv.step_result_from_flat_list(JAVA_BOARD.step(action_for_java))

    @staticmethod
    def step_result_from_flat_list(a_list):
        '''
        Convert a flat list input, a_list, to the observation, reward,
        is_done, and state dictionary.
        a_list will be a list of floats, of length (WIDTH * HEIGHT * 2 + 2).

        The first (WIDTH * HEIGHT * 2) elements of a_list represent the board state.

        The next element represents the reward, in {-1.0, 0.0, 1.0}.

        The last element represents whether the game is done, in {0.0, 1.0}.
        '''
        board_size = WIDTH * HEIGHT * 2

        obs_values = a_list[:board_size]
        # obs_values is a Py4J JavaList -> should convert to Python list
        obs = [x for x in obs_values]

        reward = a_list[board_size]

        tolerance = 0.01
        is_done = abs(a_list[board_size + 1] - 1) < tolerance

        state_dict = {'state': obs[:]}
        return obs, reward, is_done, state_dict

    def _render(self, mode='human', close=False):
        if close:
            return
        print(JAVA_BOARD.render())

'''
Dependency graph game, from attacker and defender's view.

Requirements:
    Py4J        https://www.py4j.org/download.html
    OpenAI Gym  https://github.com/openai/gym#installation
'''

from py4j.java_gateway import JavaGateway
import numpy as np
import gym

NODE_COUNT = 29
AND_NODE_COUNT = 13
EDGE_TO_OR_NODE_COUNT = 89

DEF_OBS_LENGTH = 3
ATT_OBS_LENGTH = 1

DEF_ACTION_COUNT = NODE_COUNT + 1
ATT_ACTION_COUNT = AND_NODE_COUNT + EDGE_TO_OR_NODE_COUNT + 1

DEF_INPUT_DEPTH = 2 + DEF_OBS_LENGTH * 2

DEF_OBS_SIZE = NODE_COUNT * DEF_INPUT_DEPTH
ATT_OBS_SIZE = (AND_NODE_COUNT + EDGE_TO_OR_NODE_COUNT) * 2 + NODE_COUNT * ATT_OBS_LENGTH + 1

JAVA_GAME = None
GATEWAY = None
IS_DEF_TURN = None

class DepgraphJavaEnvBoth29N(gym.Env):
    """
    Depgraph game environment.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        # https://www.py4j.org/getting_started.html
        global GATEWAY
        GATEWAY = JavaGateway()
        global JAVA_GAME
        JAVA_GAME = GATEWAY.entry_point.getGame()
        global IS_DEF_TURN
        IS_DEF_TURN = True

        self.reset()

    def _reset(self):
        result_values = JAVA_GAME.reset()
        global IS_DEF_TURN
        IS_DEF_TURN = True
        def_obs = result_values[:DEF_OBS_SIZE]
        def_obs = np.array([x for x in def_obs])
        return def_obs

    def _step(self, action):
        # action is a numpy.int64, need to convert to Python int before using with Py4J
        action_scalar = np.asscalar(action)
        action_id = action_scalar + 1

        both_obs, is_done, state_dict, is_def_turn_local = \
            DepgraphJavaEnvBoth29N.step_result_from_flat_list(
                JAVA_GAME.stepCurrent(action_id))

        global IS_DEF_TURN
        IS_DEF_TURN = is_def_turn_local

        def_obs = both_obs[:DEF_OBS_SIZE]
        att_obs = both_obs[DEF_OBS_SIZE:]

        cur_obs = def_obs
        if not IS_DEF_TURN:
            cur_obs = att_obs
        cur_obs = np.array([x for x in cur_obs])
        return cur_obs, is_done, state_dict, IS_DEF_TURN

    @staticmethod
    def step_result_from_flat_list(a_list):
        '''
        Convert a flat list input, a_list, to the observation (for defender, then
        attacker), reward, is_done, state dictionary, and is_def_turn_local.

        The first game_size elements of a_list represent the game state, first for the
        defender's view, then the attacker's.

        The next element represents the reward, in R.

        The next element represents whether the game is done, in {0.0, 1.0}.

        The last element represents whether it is the defender's turn, in {0.0, 1.0}.
        '''
        game_size = DEF_OBS_SIZE + ATT_OBS_SIZE

        both_obs = a_list[:game_size]
        # both_obs is a Py4J JavaList -> should convert to Python list
        both_obs = np.array([x for x in both_obs])

        tolerance = 0.01
        is_done = abs(a_list[game_size] - 1) < tolerance

        state_dict = {'state': both_obs[:]}

        is_def_turn_local = abs(a_list[game_size + 1] - 1) < tolerance
        return both_obs, is_done, state_dict, is_def_turn_local

    def _render(self, mode='human', close=False):
        if close:
            return
        print(JAVA_GAME.render())

    def get_defender_reward(self):
        '''
        Get the total discounted reward of the defender in the current game.
        '''
        return JAVA_GAME.getDefenderTotalPayoff()

    def get_attacker_reward(self):
        '''
        Get the total discounted reward of the attacker in the current game.
        '''
        return JAVA_GAME.getAttackerTotalPayoff()

    def get_defender_action_count(self):
        '''
        Get the number of distinct defender actions.
        '''
        return DEF_ACTION_COUNT

    def get_attacker_action_count(self):
        '''
        Get the number of distinct attacker actions.
        '''
        return ATT_ACTION_COUNT

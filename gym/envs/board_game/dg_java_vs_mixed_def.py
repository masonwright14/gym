'''
Dependency graph game.

Requirements:
    Py4J        https://www.py4j.org/download.html
    OpenAI Gym  https://github.com/openai/gym#installation
'''

import csv
import random
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters
from py4j.java_collections import ListConverter
import numpy as np
from baselines import deepq
import gym
from gym import spaces
from gym.envs.board_game.d30_def_config import DEF_MIXED_STRAT_FILE

NODE_COUNT = 30
AND_NODE_COUNT = 5
EDGE_TO_OR_NODE_COUNT = 100

DEF_OBS_LENGTH = 3
ATT_OBS_LENGTH = 1
TIME_STEPS = 10

DEF_ACTION_COUNT = NODE_COUNT + 1
ATT_ACTION_COUNT = AND_NODE_COUNT + EDGE_TO_OR_NODE_COUNT + 1

DEF_INPUT_DEPTH = 2 + DEF_OBS_LENGTH * 2

DEF_OBS_SIZE = NODE_COUNT * DEF_INPUT_DEPTH
ATT_OBS_SIZE = (AND_NODE_COUNT + EDGE_TO_OR_NODE_COUNT) * 2 + NODE_COUNT * ATT_OBS_LENGTH + 1

# DEF_MIXED_STRAT_FILE = "d30_epoch13_def.tsv"
# DEF_MIXED_STRAT_FILE = "randNoAnd_B_epoch5_def.tsv"
# DEF_MIXED_STRAT_FILE = "randNoAnd_B_epoch4_def.tsv"
# DEF_MIXED_STRAT_FILE = "randNoAnd_B_epoch3_def.tsv"
# DEF_MIXED_STRAT_FILE = "randNoAnd_B_epoch2_def.tsv"
# DEF_MIXED_STRAT_FILE = "randNoAnd_B_noNet_defStrat.tsv"
DEF_STRAT_TO_PROB = {}
IS_HEURISTIC_DEFENDER = False

JAVA_GAME = None
GATEWAY = None
IS_DEF_TURN = None

DEF_NETWORK = None
DEF_NET_NAME = None
DEF_SESS = None

MIN_PORT = 25335
ATT_PORT = None

MY_DIR = "../gym/gym/gym/envs/board_game/"

def get_lines(file_name):
    lines = None
    with open(file_name) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    lines = [x for x in lines if x]
    return lines

def read_att_port():
    port_name = MY_DIR + "d30_train_att_port.txt"
    lines = get_lines(port_name)
    port = int(lines[0])
    if port < MIN_PORT or port % 2 != 1:
        raise ValueError("Invalid att port: " + str(port))
    return port

class DepgraphJavaEnvVsMixedDef(gym.Env):
    """
    Depgraph game environment. Play against a fixed opponent.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        '''
        Set up the Java game and read in the defender mixed strategy over network and
        heuristic strategies.
        '''
        # https://www.py4j.org/getting_started.html
        global ATT_PORT
        ATT_PORT = read_att_port()
        global GATEWAY
        GATEWAY = JavaGateway(python_proxy_port=ATT_PORT,
                              gateway_parameters=GatewayParameters(port=ATT_PORT),
                              callback_server_parameters=
                              CallbackServerParameters(port=(ATT_PORT + 1)))
        global JAVA_GAME
        JAVA_GAME = GATEWAY.entry_point.getGame()

        self.setup_def_mixed_strat(DEF_MIXED_STRAT_FILE)

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
        '''
        Reset the game, draw a new defender strategy from the mixed strategy, and set this
        agent in the Java game, including whether it is a network or heuristic.
        '''
        global IS_DEF_TURN
        global IS_HEURISTIC_DEFENDER
        global DEF_NETWORK
        global DEF_NET_NAME
        global DEF_SESS

        IS_DEF_TURN = False

        cur_def_strat = self.sample_mixed_strat()
        IS_HEURISTIC_DEFENDER = DepgraphJavaEnvVsMixedDef.is_heuristic_strategy(cur_def_strat)
        is_heuristic_str = str(IS_HEURISTIC_DEFENDER)

        py_list = [is_heuristic_str, cur_def_strat]
        if not IS_HEURISTIC_DEFENDER:
            py_list = [is_heuristic_str, ""]
        java_list = ListConverter().convert(py_list, GATEWAY._gateway_client)
        result_values = JAVA_GAME.reset(java_list)
        # result_values is a Py4J JavaList -> should convert to Python list
        if IS_HEURISTIC_DEFENDER:
            def_obs = np.array([x for x in result_values])
            # def_obs = def_obs.reshape(1, def_obs.size)
            return def_obs

        cur_def_scope = self.get_net_scope(cur_def_strat)

        if cur_def_strat != DEF_NET_NAME:
            if cur_def_scope is None:
                DEF_NETWORK, _, DEF_SESS = deepq.load_for_multiple_nets(cur_def_strat)
            else:
                DEF_NETWORK, _, DEF_SESS = deepq.load_for_multiple_nets_with_scope( \
                    cur_def_strat, cur_def_scope)

            DEF_NET_NAME = cur_def_strat

        IS_DEF_TURN = True
        def_obs = DepgraphJavaEnvVsMixedDef.first_def_obs()
        att_obs, _, _, _ = self._run_def_net_until_pass(def_obs)
        return att_obs

    def _step(self, action):
        '''
        Take the attacker's action (adding a node or edge to attack set or passing).
        '''
        if IS_HEURISTIC_DEFENDER:
            return self._step_vs_heuristic(action)
        return self._step_vs_network(action)

    def _step_vs_heuristic(self, action):
        '''
        Take the attacker action (add a node or edge to attack set)
        against a heuristic defender.
        '''
        # action is a numpy.int64, need to convert to Python int before using with Py4J
        action_scalar = np.asscalar(action)
        # {1, . . ., AND_NODE_COUNT} correspond to AND nodes,
        # {AND_NODE_COUNT + 1, . . ., AND_NODE_COUNT + EDGE_TO_OR_NODE_COUNT}
        # correspond to edges to OR nodes,
        # and {AND_NODE_COUNT + EDGE_TO_OR_NODE_COUNT + 1} corresponds to "pass".
        action_id = action_scalar + 1
        return DepgraphJavaEnvVsMixedDef.step_result_from_list_heuristic( \
            JAVA_GAME.step(action_id))

    def _step_vs_network(self, action):
        '''
        Take the defender's action (adding a node to defend to the defense set or passing)
        against a network attacker.
        If the defender passes or makes an illegal move, the attacker will be allowed to
        make its move before the method returns.
        '''
        global IS_DEF_TURN

        if IS_DEF_TURN:
            raise ValueError("Must be attacker's turn here.")

        # action is a numpy.int64, need to convert to Python int before using with Py4J
        action_scalar = np.asscalar(action)
        action_id = action_scalar + 1

        both_obs, is_done, state_dict, is_def_turn_local = \
            DepgraphJavaEnvVsMixedDef.step_result_from_list_network( \
                JAVA_GAME.step(action_id))

        IS_DEF_TURN = is_def_turn_local

        if IS_DEF_TURN and not is_done:
            def_obs = both_obs[:DEF_OBS_SIZE]
            return self._run_def_net_until_pass(def_obs)

        att_obs = both_obs[DEF_OBS_SIZE:]
        att_obs = np.array([x for x in att_obs])
        # att_obs = att_obs.reshape(1, att_obs.size)

        att_reward = 0.0 # no reward for adding another item to attack set
        return att_obs, att_reward, is_done, state_dict

    def _run_def_net_until_pass(self, def_obs_arg):
        '''
        Run the game from the defender network's view, until the defender passes or
        makes an illegal move, returning control to the attacker.
        Return the attacker network's view, plus the attacker discounted marginal reward.
        '''
        global IS_DEF_TURN

        if not IS_DEF_TURN:
            raise ValueError("Must be defender's turn here.")
        if IS_HEURISTIC_DEFENDER:
            raise ValueError("Must be a network defender here.")

        att_obs = None
        def_obs = def_obs_arg
        is_done = False
        state_dict = None
        while IS_DEF_TURN and not is_done:
            def_obs = np.array([x for x in def_obs])
            def_obs = def_obs.reshape(1, def_obs.size)

            with DEF_SESS.as_default():
                raw_action = DEF_NETWORK(def_obs)
                action = raw_action[0]
                # action is a numpy.int64, need to convert to Python int,
                # before using with Py4J.
                action_scalar = np.asscalar(action)
                action_id = action_scalar + 1

                both_obs, is_done, state_dict, is_def_turn_local = \
                    DepgraphJavaEnvVsMixedDef.step_result_from_list_network( \
                        JAVA_GAME.step(action_id))

                IS_DEF_TURN = is_def_turn_local
                def_obs = both_obs[:DEF_OBS_SIZE]
                att_obs = both_obs[DEF_OBS_SIZE:]

        att_obs = np.array([x for x in att_obs])
        # att_obs = att_obs.reshape(1, att_obs.size)

        att_reward = JAVA_GAME.getSelfMarginalPayoff()
        return att_obs, att_reward, is_done, state_dict

    @staticmethod
    def is_heuristic_strategy(strategy):
        '''
        Returns True if the strategy is a network strategy, otherwise False for a heuristic.
        The name with contain ".pkl" if it is a network strategy.
        '''
        return ".pkl" not in strategy

    @staticmethod
    def step_result_from_list_heuristic(a_list):
        '''
        Convert a flat list input, a_list, to the observation, reward,
        is_done, and state dictionary.
        a_list will be a list of floats, of length (game_size + 2).

        The first game_size elements of a_list represent the game state.

        The next element represents the reward, in R.

        The last element represents whether the game is done, in {0.0, 1.0}.
        '''
        game_size = ATT_OBS_SIZE
        obs_values = a_list[:game_size]
        # obs_values is a Py4J JavaList -> should convert to Python list
        obs = np.array([x for x in obs_values])
        # obs = obs.reshape(1, obs.size)

        reward = a_list[game_size]

        tolerance = 0.01
        is_done = abs(a_list[game_size + 1] - 1) < tolerance

        state_dict = {'state': obs[:]}
        return obs, reward, is_done, state_dict

    @staticmethod
    def step_result_from_list_network(a_list):
        '''
        Convert a flat list input, a_list, to the observation, reward,
        is_done, and state dictionary.
        a_list will be a list of floats, of length (game_size + 3).

        The first game_size elements of a_list represent the game state.

        The next element represents whether the game is done, in {0.0, 1.0}.

        The last element represents whether it is the defender's turn, in {0.0, 1.0}.
        '''
        game_size = DEF_OBS_SIZE + ATT_OBS_SIZE
        obs_values = a_list[:game_size]
        # obs_values is a Py4J JavaList -> should convert to Python list
        both_obs = np.array([x for x in obs_values])

        tolerance = 0.01
        is_done = abs(a_list[game_size] - 1) < tolerance

        state_dict = {'state': both_obs[:]}

        is_def_turn_local = abs(a_list[game_size + 1] - 1) < tolerance
        return both_obs, is_done, state_dict, is_def_turn_local

    def setup_def_mixed_strat(self, strat_file):
        '''
        Load the defender's mixed strategy over heuristic and network strategies, from
        the given file.
        Should have one strategy per line, with the name and probability, tab-separated.
        '''
        global DEF_STRAT_TO_PROB
        DEF_STRAT_TO_PROB = {}
        with open(strat_file, 'r') as tsv_in:
            rows = list(list(rec) for rec in csv.reader(tsv_in, delimiter='\t'))
            for row in rows:
                if row:
                    strat = row[0]
                    prob = float(row[1])
                    if prob < 0.0 or prob > 1.0:
                        raise ValueError("Invalid prob: " + str(prob))
                    if strat in DEF_STRAT_TO_PROB:
                        raise ValueError("Duplicate strat: " + strat)
                    DEF_STRAT_TO_PROB[strat] = prob
        tol = 0.001
        if abs(sum(DEF_STRAT_TO_PROB.values()) - 1.0) > tol:
            raise ValueError("Wrong sum of probabilities: " + \
                str(sum(DEF_STRAT_TO_PROB.values())))

    def sample_mixed_strat(self):
        '''
        Draw a mixed strategy at random from the defender strategy, weighted by the
        given probabilities.
        '''
        rand_draw = random.random()
        total_prob = 0.0
        for strat, prob in DEF_STRAT_TO_PROB.items():
            total_prob += prob
            if rand_draw <= total_prob:
                return strat
        # should not get here
        return DEF_STRAT_TO_PROB.keys()[0]

    def _render(self, mode='human', close=False):
        '''
        Show human-readable view of environment state.
        '''
        if close:
            return
        print(JAVA_GAME.render())

'''
    def get_net_scope(self, net_name):
        # name is like:
        # *epochNUM_* or *epochNUM[a-z]* or *epochNUM.pkl, where NUM is an integer > 1,
        # unless "epoch" is absent, in which case return None.
        #
        # if "epoch" is absent: return None.
        # else if NUM is 2: return "deepq_train".
        # else: return "deepq_train_eNUM", inserting the integer for NUM
        if net_name == "dg_rand_30n_noAnd_B_eq_2.pkl" or \
            net_name == "dg_dqmlp_rand30NoAnd_B_att_fixed.pkl":
            return None

        epoch_index = net_name.find('epoch')
        num_start_index = epoch_index + len("epoch")

        underbar_index = net_name.find('_', num_start_index + 1)
        dot_index = net_name.find('.', num_start_index + 1)
        e_index = net_name.find('e', num_start_index + 1)
        candidates = [x for x in [underbar_index, dot_index, e_index] if x > -1]
        num_end_index = min(candidates)
        net_num = net_name[num_start_index : num_end_index]
        if net_num == "2":
            return "deepq_train"
        return "deepq_train_e" + str(net_num)
'''

    def get_net_scope(self, net_name):
        # defender name is like:
        # *_epochNUM.pkl, where NUM is an integer >= 1.
        #
        # attacker name is like:
        # *_epochNUM_att.pkl, where NUM is an integer >= 1.
        #
        # if NUM == 1: return "deepq_train"
        # else: return "deepq_train_eNUM", inserting the integer for NUM
        if "epoch1.pkl" in net_name or "epoch1_att.pkl" in net_name:
            # first round is special case: don't add _e1
            return "deepq_train"

        epoch_index = net_name.find('epoch')
        num_start_index = epoch_index + len("epoch")
        num_end_index = None
        if "_att.pkl" in net_name:
            # attacker network
            num_end_index = net_name.find("_att.pkl", num_start_index)
        else:
            # defender network
            num_end_index = net_name.find(".pkl", num_start_index)
        return "deepq_train_e" + net_name[num_start_index : num_end_index]

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

    def get_port(self):
        '''
        Get the port number used for Py4J connection.
        '''
        return ATT_PORT

    @staticmethod
    def first_def_obs():
        '''
        Get the initial observation of the defender.
        '''
        result = []
        result.extend([0.0] * NODE_COUNT)
        result.extend([TIME_STEPS * 1.0] * NODE_COUNT)
        result.extend([0.0] * (NODE_COUNT * DEF_OBS_LENGTH * 2))
        return result

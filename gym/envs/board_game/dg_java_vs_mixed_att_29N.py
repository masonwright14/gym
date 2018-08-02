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
from gym.envs.board_game.s29_att_config import ATT_MIXED_STRAT_FILE

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

# ATT_MIXED_STRAT_FILE = "sl29_randNoAndB_epoch14_att.tsv"
ATT_STRAT_TO_PROB = {}
IS_HEURISTIC_ATTACKER = False

JAVA_GAME = None
GATEWAY = None
IS_DEF_TURN = None

ATT_NETWORK = None
ATT_NET_NAME = None
ATT_SESS = None

MIN_PORT = 25333
DEF_PORT = None

MY_DIR = "../gym/gym/gym/envs/board_game/"

def get_lines(file_name):
    lines = None
    with open(file_name) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    lines = [x for x in lines if x]
    return lines

def read_def_port():
    port_name = MY_DIR + "s29_train_def_port.txt"
    lines = get_lines(port_name)
    port = int(lines[0])
    if port < MIN_PORT or port % 2 != 1:
        raise ValueError("Invalid def port: " + str(port))
    return port

class DepgraphJavaEnvVsMixedAtt29N(gym.Env):
    """
    Depgraph game environment. Play against a fixed opponent.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        '''
        Set up the Java game and read in the attacker mixed strategy over network and
        heuristic strategies.
        '''
        # https://www.py4j.org/getting_started.html
        global DEF_PORT
        DEF_PORT = read_def_port()
        global GATEWAY
        GATEWAY = JavaGateway(python_proxy_port=DEF_PORT,
                              gateway_parameters=GatewayParameters(port=DEF_PORT),
                              callback_server_parameters=
                              CallbackServerParameters(port=(DEF_PORT + 1)))
        global JAVA_GAME
        JAVA_GAME = GATEWAY.entry_point.getGame()

        self.setup_att_mixed_strat(ATT_MIXED_STRAT_FILE)

        # One action for each node or pass
        # action space is {0, . . ., NODE_COUNT}, indicating node or pass.
        self.action_space = spaces.Discrete(NODE_COUNT + 1)

        observation = self.reset()
        # convert from JavaMember object to JavaList
        observation = observation[:]
        my_shape = (len(observation),)
        self.observation_space = \
            spaces.Box(np.zeros(my_shape), np.ones(my_shape))

    def _reset(self):
        '''
        Reset the game, draw a new attacker strategy from the mixed strategy, and set this
        agent in the Java game, including whether it is a network or heuristic.
        '''
        global IS_DEF_TURN
        global IS_HEURISTIC_ATTACKER
        global ATT_NETWORK
        global ATT_NET_NAME
        global ATT_SESS

        IS_DEF_TURN = True

        cur_att_strat = self.sample_mixed_strat()
        IS_HEURISTIC_ATTACKER = DepgraphJavaEnvVsMixedAtt29N.is_heuristic_strategy(cur_att_strat)
        is_heuristic_str = str(IS_HEURISTIC_ATTACKER)

        py_list = [is_heuristic_str, cur_att_strat]
        if not IS_HEURISTIC_ATTACKER:
            py_list = [is_heuristic_str, ""]
        java_list = ListConverter().convert(py_list, GATEWAY._gateway_client)
        result_values = JAVA_GAME.reset(java_list)
        # result_values is a Py4J JavaList -> should convert to Python list
        if IS_HEURISTIC_ATTACKER:
            def_obs = np.array([x for x in result_values])
            # def_obs = def_obs.reshape(1, def_obs.size)
            return def_obs

        cur_att_scope = self.get_net_scope(cur_att_strat)

        if cur_att_strat != ATT_NET_NAME:
            if cur_att_scope is None:
                ATT_NETWORK, _, ATT_SESS = deepq.load_for_multiple_nets(cur_att_strat)
            else:
                ATT_NETWORK, _, ATT_SESS = deepq.load_for_multiple_nets_with_scope( \
                    cur_att_strat, cur_att_scope)

            ATT_NET_NAME = cur_att_strat

        def_obs = result_values[:DEF_OBS_SIZE]
        def_obs = np.array([x for x in def_obs])
        # def_obs = def_obs.reshape(1, def_obs.size)
        return def_obs

    def _step(self, action):
        '''
        Take the defender's action (adding a node to defense set or passing).
        '''
        if IS_HEURISTIC_ATTACKER:
            return self._step_vs_heuristic(action)
        return self._step_vs_network(action)

    def _step_vs_heuristic(self, action):
        '''
        Take the defender action (add a node to defense set) against a heuristic attacker.
        '''
        # action is a numpy.int64, need to convert to Python int before using with Py4J
        action_scalar = np.asscalar(action)
        # {1, . . ., NODE_COUNT} are node ids, (NODE_COUNT + 1) means "pass"
        action_id = action_scalar + 1
        return DepgraphJavaEnvVsMixedAtt29N.step_result_from_list_heuristic( \
            JAVA_GAME.step(action_id))

    def _step_vs_network(self, action):
        '''
        Take the defender's action (adding a node to defend to the defense set or passing)
        against a network attacker.
        If the defender passes or makes an illegal move, the attacker will be allowed to
        make its move before the method returns.
        '''
        global IS_DEF_TURN

        if not IS_DEF_TURN:
            raise ValueError("Must be defender's turn here.")

        # action is a numpy.int64, need to convert to Python int before using with Py4J
        action_scalar = np.asscalar(action)
        action_id = action_scalar + 1

        both_obs, is_done, state_dict, is_def_turn_local = \
            DepgraphJavaEnvVsMixedAtt29N.step_result_from_list_network( \
                JAVA_GAME.step(action_id))

        IS_DEF_TURN = is_def_turn_local

        if not IS_DEF_TURN and not is_done:
            att_obs = both_obs[DEF_OBS_SIZE:]
            return self._run_att_net_until_pass(att_obs)

        def_obs = both_obs[:DEF_OBS_SIZE]
        def_obs = np.array([x for x in def_obs])
        # def_obs = def_obs.reshape(1, def_obs.size)

        def_reward = 0.0 # no reward for adding another item to defense set
        return def_obs, def_reward, is_done, state_dict

    def _run_att_net_until_pass(self, att_obs_arg):
        '''
        Run the game from the attacker network's view, until the attacker passes or
        makes an illegal move, returning control to the defender.
        Return the defender network's view, plus the defender discounted marginal reward.
        '''
        global IS_DEF_TURN

        if IS_DEF_TURN:
            raise ValueError("Must be attacker's turn here.")
        if IS_HEURISTIC_ATTACKER:
            raise ValueError("Must be a network attacker here.")

        def_obs = None
        att_obs = att_obs_arg
        is_done = False
        state_dict = None
        while not IS_DEF_TURN and not is_done:
            att_obs = np.array([x for x in att_obs])
            att_obs = att_obs.reshape(1, att_obs.size)

            with ATT_SESS.as_default():
                raw_action = ATT_NETWORK(att_obs)
                action = raw_action[0]
                # action is a numpy.int64, need to convert to Python int,
                # before using with Py4J.
                action_scalar = np.asscalar(action)
                action_id = action_scalar + 1

                both_obs, is_done, state_dict, is_def_turn_local = \
                    DepgraphJavaEnvVsMixedAtt29N.step_result_from_list_network( \
                        JAVA_GAME.step(action_id))

                IS_DEF_TURN = is_def_turn_local
                def_obs = both_obs[:DEF_OBS_SIZE]
                att_obs = both_obs[DEF_OBS_SIZE:]

        def_obs = np.array([x for x in def_obs])
        # def_obs = def_obs.reshape(1, def_obs.size)

        def_reward = JAVA_GAME.getSelfMarginalPayoff()
        return def_obs, def_reward, is_done, state_dict

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
        a_list will be a list of floats, of length (NODE_COUNT * DEF_INPUT_DEPTH+ 2).

        The first (NODE_COUNT * DEF_INPUT_DEPTH) elements of a_list represent the game state.

        The next element represents the reward, in R-.

        The last element represents whether the game is done, in {0.0, 1.0}.
        '''
        game_size = NODE_COUNT * DEF_INPUT_DEPTH

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

        # edit DepgraphPy4JDefVsNetOrHeuristic to return this.
        tolerance = 0.01
        is_done = abs(a_list[game_size] - 1) < tolerance

        state_dict = {'state': both_obs[:]}

        is_def_turn_local = abs(a_list[game_size + 1] - 1) < tolerance
        return both_obs, is_done, state_dict, is_def_turn_local

    def setup_att_mixed_strat(self, strat_file):
        '''
        Load the attacker's mixed strategy over heuristic and network strategies, from
        the given file.
        Should have one strategy per line, with the name and probability, tab-separated.
        '''
        global ATT_STRAT_TO_PROB
        ATT_STRAT_TO_PROB = {}
        with open(strat_file, 'r') as tsv_in:
            rows = list(list(rec) for rec in csv.reader(tsv_in, delimiter='\t'))
            for row in rows:
                if row:
                    strat = row[0]
                    prob = float(row[1])
                    if prob < 0.0 or prob > 1.0:
                        raise ValueError("Invalid prob: " + str(prob))
                    if strat in ATT_STRAT_TO_PROB:
                        raise ValueError("Duplicate strat: " + strat)
                    ATT_STRAT_TO_PROB[strat] = prob
        tol = 0.001
        if abs(sum(ATT_STRAT_TO_PROB.values()) - 1.0) > tol:
            raise ValueError("Wrong sum of probabilities: " + \
                str(sum(ATT_STRAT_TO_PROB.values())))

    def sample_mixed_strat(self):
        '''
        Draw a mixed strategy at random from the attacker strategy, weighted by the
        given probabilities.
        '''
        rand_draw = random.random()
        total_prob = 0.0
        for strat, prob in ATT_STRAT_TO_PROB.items():
            total_prob += prob
            if rand_draw <= total_prob:
                return strat
        # should not get here
        return ATT_STRAT_TO_PROB.keys()[0]

    def update_for_retrain(self, strat_file):
        '''
        Change to the given opponent mixed strategy and reset game.
        '''
        self.setup_att_mixed_strat(strat_file)
        self.reset()

    def _render(self, mode='human', close=False):
        '''
        Show human-readable view of environment state.
        '''
        if close:
            return
        print(JAVA_GAME.render())

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
        Get the total discounted reward of the opponent (attacker) in the current game.
        '''
        return JAVA_GAME.getOpponentTotalPayoff()

    def get_self_reward(self):
        '''
        Get the total discounted reward of self (defender) in the current game.
        '''
        return JAVA_GAME.getSelfTotalPayoff()

    def get_port(self):
        '''
        Get the port number used for Py4J connection.
        '''
        return DEF_PORT

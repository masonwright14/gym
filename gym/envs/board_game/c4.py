'''
Game of Connect Four
'''

from six import StringIO
import sys
import gym
from gym import spaces
import numpy as np
from gym import error
from gym.utils import seeding

HEIGHT = 6
WIDTH = 7
GOAL = 4
RED = 0
BLACK = 1
MAX_VALUE = 100
SEARCH_DEPTH = 1
MAX_SEARCH_DEPTH = 5

def make_human_policy():
    ''' Create the policy where the agent plays against human opponent. '''
    def show_board(board):
        ''' Display the board for human to choose move. '''
        my_str = ''
        for row in reversed(range(HEIGHT)):
            for col in range(WIDTH):
                if board[2, row, col] == 1:
                    my_str += '_\t'
                elif board[RED, row, col] == 1:
                    my_str += 'O\t'
                else:
                    my_str += 'X\t'
            my_str += '\n'
            my_str += '\n'
        print(my_str)

    def human_policy(state):
        '''
        Return None if no legal moves left, else prompt the user for a move,
        as the integer column name. Wait until a valid move is entered, then return it.
        '''
        move = -1
        possible_moves = C4Env.get_possible_actions(state)
        # No moves left
        if not possible_moves:
            return None

        show_board(state)
        highest_move = WIDTH - 1
        while move not in possible_moves:
            text = input("Enter move, 0 to " + str(highest_move) + ": ")
            try:
                move = int(text)
                if move not in possible_moves:
                    print("Not a legal move. Try: " + str(possible_moves))
            except ValueError:
                print('Please enter an integer')
        return move
    return human_policy

def make_random_policy(np_random):
    ''' Create the random policy action function. '''
    def random_policy(state):
        ''' Return a random legal action, if any is avilable, else None. '''
        possible_moves = C4Env.get_possible_actions(state)
        # No moves left
        if not possible_moves:
            return None
        index = np_random.randint(len(possible_moves))
        return possible_moves[index]
    return random_policy

def make_minimax_policy(np_random):
    ''' Create the minimax policy action function. '''
    def minimax_policy(state):
        ''' Return the minimax-optimal action if any action is available, else None. '''
        possible_moves = C4Env.get_possible_actions(state)
        if not possible_moves:
            return None
        if is_empty(state):
            return WIDTH // 2 # move in center if first move
        cols = np.asarray(possible_moves)
        np_random.shuffle(cols)
        best_move = -1
        max_value = -MAX_VALUE - 1
        for col in np.nditer(cols):
            if SEARCH_DEPTH == 0:
                return col
            board_copy = np.copy(state)
            C4Env.make_move(board_copy, col, RED)
            cur_value = minimax_policy_recurse(board_copy, SEARCH_DEPTH - 1)
            if cur_value > max_value:
                max_value = cur_value
                best_move = col
        return best_move

    def minimax_policy_recurse(board, depth_left):
        '''
        board: the 3 * HEIGHT * WIDTH board state, elements in {0, 1}
        depth_left the nonnegative integer depth to search further

        result: the value of the board for RED player, based on minimax search.
        will be up to MAX_VALUE if red wins, down to -MAX_VALUE if black wins.
        '''
        winner = C4Env.game_finished(board)
        if winner == 1: # RED wins
            return MAX_VALUE
        if winner == -1: # BLACK wins
            return -1 * MAX_VALUE
        if C4Env.is_game_over(board): # draw
            return 0
        if depth_left == 0: # return benefit for red (self)
            return board_value(board)
        legal_moves = C4Env.get_possible_actions(board)
        if is_black_turn(board):
            # minimizing player (opponent)
            min_value = MAX_VALUE
            for col in legal_moves:
                board_copy = np.copy(board)
                C4Env.make_move(board_copy, col, BLACK)
                cur_value = minimax_policy_recurse(board_copy, depth_left - 1)
                if cur_value < min_value:
                    min_value = cur_value
            return min_value
        # maximizing player (self)
        max_value = -1 * MAX_VALUE
        for col in legal_moves:
            board_copy = np.copy(board)
            C4Env.make_move(board_copy, col, RED)
            cur_value = minimax_policy_recurse(board_copy, depth_left - 1)
            if cur_value > max_value:
                max_value = cur_value
        return max_value

    def is_empty(board):
        ''' No  moves have been made yet. '''
        return np.sum(board[0]) == 0

    def is_black_turn(board):
        ''' Return True if it is second player (BLACK) turn to move. '''
        return np.sum(board[0]) > np.sum(board[1])

    def board_value(board):
        ''' Returns value of board estimated for RED (firts player).
        Will be MAX_VALUE if red won, -MAX_VALUE if black won,
        else the difference between the longest row for red and black.
        '''
        winner = C4Env.game_finished(board)
        if winner == 1: # RED wins
            return MAX_VALUE
        if winner == -1: # BLACK wins
            return -1 * MAX_VALUE
        if C4Env.is_game_over(board): # draw
            return 0
        red_length = max_color_length(board[0])
        black_length = max_color_length(board[1])
        return red_length - black_length

    def max_color_length(board_slice):
        '''
        Returns the length of the longest run, for one color's slice of the board.
        '''
        result = 0
        # check vertical
        for col in range(WIDTH):
            color_length = 0
            for row in range(HEIGHT):
                if board_slice[row, col] == 1:
                    color_length += 1
                    if color_length > result:
                        result = color_length
                else:
                    color_length = 0

        # check horizontal
        for row in range(HEIGHT):
            color_length = 0
            for col in range(WIDTH):
                if board_slice[row, col] == 1:
                    color_length += 1
                    if color_length > result:
                        result = color_length
                else:
                    color_length = 0

        # check for right diagonal line
        diagonals = 6
        for i in range(diagonals):
            top = min(GOAL + i - 1, HEIGHT - 1)
            left = max(i - 2, 0)
            row = top
            col = left
            color_length = 0
            while row >= 0 and col < WIDTH:
                if board_slice[row, col] == 1:
                    color_length += 1
                    if color_length > result:
                        result = color_length
                else:
                    color_length = 0
                row -= 1
                col += 1

        # check for left diagonal line
        for i in range(diagonals):
            top = min(GOAL + i - 1, HEIGHT - 1)
            right = min(WIDTH - i + 1, WIDTH - 1)
            row = top
            col = right
            color_length = 0
            while row >= 0 and col >= 0:
                if board_slice[row, col] == 1:
                    color_length += 1
                    if color_length > result:
                        result = color_length
                else:
                    color_length = 0
                row -= 1
                col -= 1
        return result
    return minimax_policy

class C4Env(gym.Env):
    """
    Connect Four environment. Play against a fixed opponent.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, player_color, opponent, observation_type, illegal_move_mode, \
        obs_type, search_depth):
        """
        Args:
            opponent: An opponent policy
            illegal_move_mode: What to do when the agent makes an illegal move.
            Choices: 'raise' or 'lose'
        """
        colormap = {
            'black': BLACK,
            'red': RED,
        }
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error( \
                "player_color must be 'black' or 'red', not {}".format(player_color))

        if search_depth < 0 or search_depth > MAX_SEARCH_DEPTH:
            raise ValueError("Illegal search depth: " + str(search_depth))
        global SEARCH_DEPTH
        SEARCH_DEPTH = search_depth

        self.opponent = opponent

        assert observation_type in ['numpy3c']
        self.observation_type = observation_type

        assert illegal_move_mode in ['lose', 'raise']
        self.illegal_move_mode = illegal_move_mode

        if self.observation_type != 'numpy3c':
            raise error.Error( \
                'Unsupported observation type: {}'.format(self.observation_type))

        # One action for each board position or resign
        # action space is {0, . . ., WIDTH}, indicating column or resign
        self.action_space = spaces.Discrete(WIDTH)

        obs_types = ['vector', 'tensor', 'tensor_aug']
        if obs_type not in obs_types:
            raise ValueError("Unrecognized obs_type: " + str(obs_type))
        self.is_vector = (obs_type == 'vector')
        self.is_augmented = (obs_type == 'tensor_aug')

        self._seed()
        observation = self.reset()
        self.observation_space = \
            spaces.Box(np.zeros(observation.shape), np.ones(observation.shape))


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == 'random':
                self.opponent_policy = make_random_policy(self.np_random)
            elif self.opponent == 'minimax':
                self.opponent_policy = make_minimax_policy(self.np_random)
            elif self.opponent == 'human':
                self.opponent_policy = make_human_policy()
            else:
                raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent
        return [seed]

    def _reset(self):
        self.state = np.zeros((3, HEIGHT, WIDTH))
        self.state[2, :, :] = 1.0
        self.to_play = RED
        self.done = False

        # Let the opponent play if it's not the agent's turn
        if self.player_color != self.to_play:
            index = self.opponent_policy(self.state)
            C4Env.make_move(self.state, index, RED)
            self.to_play = BLACK
        return self.get_obs()

    def get_obs(self):
        '''
        If self.is_vector, return the flattened (vector) version of the state.
        Otherwise, return the unalterered, tensor version of the state.
        But if is_augmented is true, augment the tensor version with a plane
        that has all zeros in columns that are not legal moves, all ones in
        columns that are legal moves.
        '''
        if self.is_vector:
            return C4Env.as_vector(self.state)
        if not self.is_augmented:
            return self.state
        result = np.zeros((4, HEIGHT, WIDTH))
        result[1:, :, :] = self.state
        possible_moves = C4Env.get_possible_actions(self.state)
        for col in possible_moves:
            result[0, :, col] = 1
        return result

    def _step(self, action):
        assert self.to_play == self.player_color
        # If already terminal, then don't do anything
        if self.done:
            return self.get_obs(), 0., True, {'state': self.state}

        if C4Env.resign_move(action):
            return self.get_obs(), -1, True, {'state': self.state}
        elif not C4Env.valid_move(self.state, action):
            if self.illegal_move_mode == 'raise':
                raise
            elif self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = True
                return self.get_obs(), -1., True, {'state': self.state}
            else:
                raise error.Error( \
                    'Unsupported illegal move action: {}'.format(self.illegal_move_mode))
        else:
            C4Env.make_move(self.state, action, self.player_color)

        # Opponent play
        index = self.opponent_policy(self.state)

        # Making move if there are moves left
        if index is not None:
            if C4Env.resign_move(index):
                return self.get_obs(), 1, True, {'state': self.state}
            else:
                C4Env.make_move(self.state, index, 1 - self.player_color)

        reward = C4Env.game_finished(self.state)
        if self.player_color == BLACK:
            reward = - reward
        self.done = reward != 0
        return self.get_obs(), reward, self.done, {'state': self.state}

    @staticmethod
    def is_game_over(board):
        # game is over if someone has four in a row, or no legal moves exist
        return C4Env.game_finished(board) != 0 or \
            len(C4Env.get_possible_actions(board)) == 0

    @staticmethod
    def get_possible_actions(board):
        if C4Env.game_finished(board) != 0:
            return []
        return [x for x in range(WIDTH) if board[2, HEIGHT - 1, x] == 1]

    @staticmethod
    def resign_move(action):
        return action == WIDTH

    @staticmethod
    def make_move(board, action, player):
        for i in range(HEIGHT):
            if board[2, i, action] == 1:
                board[2, i, action] = 0
                board[player, i, action] = 1
                return
        raise ValueError("Illegal move: " + str(action))

    @staticmethod
    def as_vector(board):
        return np.reshape(board, 3 * HEIGHT * WIDTH)

    @staticmethod
    def valid_move(board, action):
        return board[2, HEIGHT - 1, action] == 1

    @staticmethod
    def game_finished(board):
        # Returns 1 if player RED wins, -1 if player BLACK wins and 0 otherwise
        # check vertical
        for col in range(WIDTH):
            red_length = 0
            black_length = 0
            for row in range(HEIGHT):
                if board[RED, row, col] == 1:
                    red_length += 1
                    if red_length == GOAL:
                        return 1
                    black_length = 0
                elif board[BLACK, row, col] == 1:
                    black_length += 1
                    if black_length == GOAL:
                        return -1
                    red_length = 0
                else: # empty cells above, skip column
                    continue

        # check horizontal
        for row in range(HEIGHT):
            red_length = 0
            black_length = 0
            for col in range(WIDTH):
                if board[RED, row, col] == 1:
                    red_length += 1
                    if red_length == GOAL:
                        return 1
                    black_length = 0
                elif board[BLACK, row, col] == 1:
                    black_length += 1
                    if black_length == GOAL:
                        return -1
                    red_length = 0
                else:
                    red_length = 0
                    black_length = 0

        # check for right diagonal line
        diagonals = 6
        for i in range(diagonals):
            top = min(GOAL + i - 1, HEIGHT - 1)
            left = max(i - 2, 0)
            row = top
            col = left
            red_length = 0
            black_length = 0
            while row >= 0 and col < WIDTH:
                if board[RED, row, col] == 1:
                    red_length += 1
                    if red_length == GOAL:
                        return 1
                    black_length = 0
                elif board[BLACK, row, col] == 1:
                    black_length += 1
                    if black_length == GOAL:
                        return -1
                    red_length = 0
                else:
                    red_length = 0
                    black_length = 0
                row -= 1
                col += 1

        # check for left diagonal line
        for i in range(diagonals):
            top = min(GOAL + i - 1, HEIGHT - 1)
            right = min(WIDTH - i + 1, WIDTH - 1)
            row = top
            col = right
            red_length = 0
            black_length = 0
            while row >= 0 and col >= 0:
                if board[RED, row, col] == 1:
                    red_length += 1
                    if red_length == GOAL:
                        return 1
                    black_length = 0
                elif board[BLACK, row, col] == 1:
                    red_length = 0
                    black_length += 1
                    if black_length == GOAL:
                        return -1
                else:
                    red_length = 0
                    black_length = 0
                row -= 1
                col -= 1

        # no four in a row
        return 0

    def _render(self, mode='human', close=False):
        if close:
            return
        board = self.state
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for row in reversed(range(HEIGHT)):
            for col in range(WIDTH):
                if board[2, row, col] == 1:
                    outfile.write('_\t')
                elif board[RED, row, col] == 1:
                    outfile.write('O\t')
                else:
                    outfile.write('X\t')
            outfile.write('\n')
            outfile.write('\n')

        if mode != 'human':
            return outfile

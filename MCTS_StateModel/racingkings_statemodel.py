# from Interface import TensorNotation
from Interpreter import game
from copy import deepcopy
from mcts import mcts

STD_FEN = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"


class RacingKingsState():
    def __init__(self, board=STD_FEN):
        self.game = game.Game()     #very ugly... Instances: 1 attribute, 2 import, 3 constructor
        self.game.player_to_move = 1
        # self.board = board
        self.currentPlayer = 1

    # def __eq__(self, other):
        # raise NotImplementedError()
        # self

# TODO: mcts expects this to be -1 or 1. game() expects this to be 0 or 1.
    def getCurrentPlayer(self):
        return self.currentPlayer

    def getPossibleActions(self):
        possible_actions = []
        moves = self.game.get_move_list()
        for m in moves:
            possible_actions.append(Action(self, m))
        return possible_actions

    def takeAction(self, action):
        new_state = deepcopy(self)
        # new_state.game.board = action.move   #TODO: right format?
        # new_state.game.board.push(action.move)  #TODO: maybe this is better?
        new_state.game.make_move(action.move)
        # new_state.game.player_to_move = self.game.player_to_move * -1
        new_state.currentPlayer = self.currentPlayer * -1
        return new_state


    def isTerminal(self):
        return self.game.is_ended()

    def getReward(self):
        # only needed for terminal states
        if self.game.is_won():
            return 1
        elif self.game.is_draw():
            return 0
        elif self.game.is_ended():
            return -1
        return False

    def show_board(self, path=None):
        self.game.show_game()


class Action():
    def __init__(self, player, move):
        self.player = player
        self.move = move

    def __str__(self):
        return str(self.move)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.move == other.move and self.player == other.player

    def __hash__(self):
        return hash((self.player, self.move))

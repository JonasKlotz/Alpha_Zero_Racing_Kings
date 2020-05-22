import aztsTree as at
import stateMachine as sm
import mockModel

from config import *


class Player():
    def __init__(self, color, runs_per_move = RUNS_PER_MOVE):
        self.state_machine = sm.StateMachine()
        self.model = mockModel.MockModel()
        self.tree = at.AztsTree(self.state_machine, \
                              self.model, \
                              color, \
                              None, \
                              runs_per_move)

    def make_move(self):
        return self.tree.make_move()

    def receive_move(self, move):
        return self.tree.receive_move(move)

    def dump_data(self):
        return [self.tree.get_position(), \
                self.tree.get_policy_tensor(), \
                None]


if __name__ == "__main__":
    player = Player(WHITE)
    print(f"First move of white player is {player.make_move()}.")

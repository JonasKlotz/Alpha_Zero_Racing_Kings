"""
This module simulates games of RacingsKings.
"""

import os.path
import time
import pickle
import azts
from Interpreter import game
import state_machine as sm
import screen
import config

WHITE = 1
BLACK = -1
REPORT_CYCLE = 25


class Player():
    def __init__(self, color, runs_per_move=100):
        self.state_machine = sm.StateMachine()
        self.model = azts.MockModel()
        self.tree = azts.Azts(self.state_machine, \
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


class SelfMatch():
    def __init__(self):
        self.p1 = Player(WHITE, config.RUNS_PER_MOVE)
        self.p2 = Player(BLACK)
        self.game = game.Game()
        self.screen = screen.Screen()
        self.data_collection = []

    def simulate(self):
        moves = 0
        time1 = time.time()
        while not self.game.is_ended():

            if moves % REPORT_CYCLE == 0:
                time2 = time.time()
                elapsed = time2 - time1
                avg_per_move = elapsed / REPORT_CYCLE
                print(f"played {moves} moves in {str(elapsed)[0:5]} " \
                      + f"seconds, average of {str(avg_per_move)[0:4]} " \
                      + f"second per move.")
                time1 = time.time()

            moves += 1
            white_move = self.p1.make_move()
            self.game.make_move(white_move)
            self.data_collection.append(self.p1.dump_data())

            if config.SHOW_GAME:
                img = self.game.render_game()
                self.screen.show_img(img)

            if self.game.is_ended():
                break

            self.p2.receive_move(white_move)
            black_move = self.p2.make_move()
            self.game.make_move(black_move)
            self.data_collection.append(self.p2.dump_data())

            if config.SHOW_GAME:
                img = self.game.render_game()
                self.screen.show_img(img)

            self.p1.receive_move(black_move)

        result = self.game.board.result()
        print(f"game ended after {moves} " \
              + f"moves with {result}.")
        translate = {"*": 0, "1-0": 1, "0-1": -1, "1/2-1/2": 0}
        result = translate[result]
        for i in self.data_collection:
            i[2] = result


class SelfPlay():
    def __init__(self):
        self.match = SelfMatch()

    def start(self, iterations=100):
        for i in range(iterations):
            self.match.simulate()
            data = [tuple(j) for j in self.match.data_collection]

            filenumber = i

            filenumberstring = str(filenumber).zfill(4)
            filename = f"game_{filenumberstring}.pkl"
            while os.path.isfile(filename):
                filenumber += 1
                filenumberstring = str(filenumber).zfill(4)
                filename = f"game_{filenumberstring}.pkl"

            pickle.dump(data, open(config.GAMEDIR + "/" + filename, "wb"))

            del self.match
            self.match = SelfMatch()


if __name__ == "__main__":
    play = SelfPlay()
    play.start(50)

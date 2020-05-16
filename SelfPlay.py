import azts
from Interpreter import game
import StateMachine as sm
import Screen

WHITE = 1
BLACK = -1
SHOW_GAME = True
RUNS_PER_MOVE = 10

class Player():
    def __init__(self, color, num_of_runs = 100):
        self.state_machine = sm.StateMachine()
        self.model = azts.MockModel()
        self.tree = azts.Azts(self.state_machine, \
                self.model, \
                color, \
                None, \
                num_of_runs)
    
    def make_move(self):
        return self.tree.make_move()

    def receive_move(self, move):
        return self.tree.receive_move(move)

    def dump_data(self):
        return (self.tree.get_position(), \
                self.tree.get_policy_tensor())




class SelfMatch():
    def __init__(self):
        self.p1 = Player(WHITE, RUNS_PER_MOVE)
        self.p2 = Player(BLACK)
        self.game = game.Game() 
        self.screen = Screen.Screen()

    def simulate(self):
        moves = 0
        while not self.game.is_ended():
            if moves % 25 == 0:
                print(f"played {moves} moves.")
            moves += 1
            white_move = self.p1.make_move()
            self.game.make_move(white_move)

            if SHOW_GAME:
                img = self.game.render_game()
                self.screen.show_img(img)


            if self.game.is_ended():
                break 


            self.p2.receive_move(white_move)
            black_move = self.p2.make_move()
            self.game.make_move(black_move)

            if SHOW_GAME:
                img = self.game.render_game()
                self.screen.show_img(img)

            self.p1.receive_move(black_move)

        result = self.game.board.result()
        print(f"game ended after {moves} " \
                + f"moves with {result}.")


if __name__ == "__main__":
    match = SelfMatch()
    match.simulate()



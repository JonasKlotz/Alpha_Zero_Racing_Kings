import azts
from Interpreter import game
import StateMachine as sm


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

    def receive_move(self):
        return self.tree.receive_move()

    def dump_data(self):
        return (self.tree.get_position() , tree.get_policy_tensor())




class SelfPlay():
    def __init__():
        pass

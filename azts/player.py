
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

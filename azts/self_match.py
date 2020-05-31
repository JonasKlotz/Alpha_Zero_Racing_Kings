# pylint: disable=E0401
# pylint: disable=E0602
'''
Self match puts two ai players
in a match against each other
'''
import time

from Interpreter import game
from Player import config
from azts import mock_model
from azts import player
from azts import screen
from azts.config import ROLLOUT_PAYOFFS, \
        EXPLORATION, HEAT, BLACK, WHITE, \
        RUNS_PER_MOVE, TO_STRING, TRAINING_PAYOFFS, \
        SHOW_GAME, DEFAULT_PLAYER

REPORT_CYCLE = 10

class SelfMatch():
    '''
    Self match puts two ai players in
    a match against each other and
    collects the respective move distribution
    of the players for each position. At the
    end of the match, the data collection
    is annotated with the outcome of the
    game.
    Initialise SelfMatch with the number
    of rollouts that each player does per
    move
    '''
    def __init__(self, \
            player_one, \
            player_two, \
            runs_per_move=RUNS_PER_MOVE):

        self.players = []

        player_one.set_color(WHITE)
        self.players.append(player_one)
        player_two.set_color(BLACK)
        self.players.append(player_two)

        self.game = game.Game()
        self.screen = screen.Screen()
        self.data_collection = []

        self.training_payoffs = player_one["training_payoffs"] \
                if "training_payoffs" in player_one.keys() \
                else TRAINING_PAYOFFS

    def set_game_state(self, fen_state):
        '''
        set game state of both players to a
        state provided with a fen string
        :param str fen_state: the state to set
        the two players to.
        '''
        _ = [i.set_game_state(fen_state) for i in self.players]
        self.game.board.set_fen(fen_state)

    def simulate(self):
        '''
        simulate a game. this starts a
        loop of taking turns and making
        moves between the players while
        storing each game position and
        corresponding move distributions
        in data collection. loop ends with
        end of match.
        :return int: state in which game
        ended according to enum type
        defined in config: running, white
        wins, black wins, draw, draw by
        stale mate, draw by repetition,
        draw by two wins
        '''
        moves = 1
        time1 = time.time()
        while True:
            # check break condition:
            if self.game.is_ended():
                break
            # select players
            select = 0 if self.game.get_current_player() else 1
            active_player = self.players[select]
            other_player = self.players[1 - select]
            # handle all moves
            move = active_player.make_move()
            other_player.receive_move(move)
            self.game.make_move(move)
            # collect data
            self.data_collection.append(active_player.dump_data())

            # statistics:
            # only increment after black move
            moves += select
            self._show_game()
            if moves % REPORT_CYCLE == 0 and select:
                time1 = self._report(time1, moves)

        result = self.game.board.result()
        state = self.game.get_game_state()
        print(f"game ended after {moves} " \
              + f"moves with {result} ({TO_STRING[state]}).")
        score = self.training_payoffs[state]

        for i in self.data_collection:
            i[2] = score

        return state


    def _show_game(self):
        if SHOW_GAME:
            img = self.game.render_game()
            self.screen.show_img(img)

    def _report(self, time_before, moves):
        time_now = time.time()
        elapsed = time_now - time_before
        avg_per_move = elapsed / REPORT_CYCLE
        print(f"total moves: {moves}; {REPORT_CYCLE} moves in " \
                + f"{str(elapsed)[0:5]}s, average of " \
                + f"{str(avg_per_move)[0:4]}s per move.")
        return time_now

if __name__ == "__main__":
    SHOW_GAME = True
    RUNS_PER_MOVE = 10

    model = mock_model.Model()
    conf_player_one = config("Player/default_config.yaml")



    match = SelfMatch()
    match.simulate()

# pylint: enable=E0401
# pylint: enable=E0602

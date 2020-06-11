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
        SHOW_GAME

from lib.logger import get_logger
log = get_logger("SelfMatch")

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

    def __init__(self,
                 player_one,
                 player_two,
                 runs_per_move=RUNS_PER_MOVE,
                 show_game=SHOW_GAME,
                 report_cycle=REPORT_CYCLE,
                 track_player=WHITE):

        self.players = [player_one, player_two]

        for i, j in zip(self.players, [WHITE, BLACK]):
            i.set_color(j)
            i.set_runs_per_move(runs_per_move) 

        self.game = game.Game()
        self.screen = screen.Screen()
        self.data_collection = []

        self.training_payoffs = TRAINING_PAYOFFS
        self.show_game = show_game
        self.report_cycle = report_cycle
        self.match_moves = []
        self.track_player = track_player

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
        log.info(f"\nWHITE: {self.players[0].name}\n"
              + f"BLACK: {self.players[1].name}\n")
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
            if moves % self.report_cycle == 0 and ~select:
                time1 = self._report(time1, moves)

        result = self.game.board.result()
        state = self.game.get_game_state()
        log.info(f"game ended after {moves} "
              + f"moves with {result} ({TO_STRING[state]}).")
        score = self.training_payoffs[state]

        for i in self.data_collection:
            i[2] = score

        return state

    def _show_game(self):
        if self.show_game:
            img = self.game.render_game()
            self.screen.show_img(img)

    def _report(self, time_before, moves):
        time_now = time.time()
        elapsed = time_now - time_before
        avg_per_move = elapsed / self.report_cycle
        log.info(f"total moves: {moves}; {self.report_cycle} moves in "
              + f"{str(elapsed)[0:5]}s, average of "
                + f"{str(avg_per_move)[0:4]}s per move.")
        return time_now


if __name__ == "__main__":
    SHOW_GAME = True
    RUNS_PER_MOVE = 10

    model = mock_model.MockModel()

    players = {}
    for i, j in zip(["player_one", "player_two"],
                    ["default_config.yaml", "StockingFish.yaml"]):
        path = "Player/" + j
        configuration = config.Config(path)
        players[i] = player.Player(model=model,
                                   name=configuration.name,
                                   **(configuration.player.as_dictionary()))

    match = SelfMatch(**players)
    match.simulate()

# pylint: enable=E0401
# pylint: enable=E0602

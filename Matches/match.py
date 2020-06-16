# pylint: disable=E0401
# pylint: disable=E0602
'''
match puts two ai players
in a match against each other
'''
import time
import os
import sys
from lib.logger import get_logger

from Interpreter import game
from Player import config
from Azts import mock_model
from Azts import player
from Azts import screen
from Azts import utility
from Azts.config import ROLLOUT_PAYOFFS, \
        EXPLORATION, HEAT, BLACK, WHITE, \
        ROLLOUTS_PER_MOVE, TO_STRING, TRAINING_PAYOFFS, \
        SHOW_GAME

log = get_logger("Match")

REPORT_CYCLE = 10


class Match():
    '''
    Match puts two ai players in
    a match against each other and
    collects the respective move distribution
    of the players for each position. At the
    end of the match, the data collection
    is annotated with the outcome of the
    game.
    Initialise Match with the number
    of rollouts that each player does per
    move
    '''

    def __init__(self,
                 player_one,
                 player_two,
                 rollouts_per_move=ROLLOUTS_PER_MOVE,
                 show_game=SHOW_GAME,
                 report_cycle=REPORT_CYCLE,
                 track_player=WHITE):

        self.players = [player_one, player_two]

        for i, j in zip(self.players, [WHITE, BLACK]):
            i.set_color(j)
            i.set_rollouts_per_move(rollouts_per_move) 

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
        log.info(f"\n\nin process {os.getpid()}:\n" \
              + f"WHITE: {self.players[0].name}\n"
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
            move = self._handle_user_input(move, select)

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

        return self._clean_up_end_game(moves)


    def _handle_user_input(self, move, select):
        if move == "exit":
            for i in self.players:
                i.stop()
            sys.exit()
        if move == "tree":
            print(self.players[1 - select].tree)
            move = self.players[select].make_move()

        return move

    def _clean_up_end_game(self, moves):
        '''
        collect results, shut down players and
        return state
        :param int moves: number of moves played,
        just to display that to the logger
        '''
        result = self.game.board.result()
        state = self.game.get_game_state()
        log.info(f"game ended after {moves} "
              + f"moves with {result} ({TO_STRING[state]}).")
        score = self.training_payoffs[state]

        for i in self.players:
            i.stop()

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
        log.info(f"process {os.getpid()}: total moves: {moves}; " \
                + f"{self.report_cycle} moves in " \
                + f"{str(elapsed)[0:5]}s, average of " \
                + f"{str(avg_per_move)[0:4]}s per move.")
        return time_now


if __name__ == "__main__":
    SHOW_GAME = True
    ROLLOUTS_PER_MOVE = 10

    model = mock_model.MockModel()

    players = {}
    for i, j in zip(["player_one", "player_two"],
                    ["default_config.yaml", "StockingFish.yaml"]):
        players[i] = utility.load_player(j)

    match = Match(**players)
    match.simulate()

# pylint: enable=E0401
# pylint: enable=E0602

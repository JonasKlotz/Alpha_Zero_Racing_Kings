import time

from azts import player
from azts import screen
from Interpreter import game

from azts.config import WHITE, BLACK, RUNS_PER_MOVE, SHOW_GAME

REPORT_CYCLE = 10


class SelfMatch():
    def __init__(self):
        self.players = []
        self.players.append(player.Player(WHITE, RUNS_PER_MOVE))
        self.players.append(player.Player(BLACK, RUNS_PER_MOVE))
        self.game = game.Game()
        self.screen = screen.Screen()
        self.data_collection = []

    def set_game_state(self, fen_state):
        _ = [i.set_game_state(fen_state) for i in self.players]
        self.game.board.set_fen(fen_state)

    def simulate(self):
        moves = 1
        time1 = time.time()
        while True:
            # check break condition: 
            if self.game.is_ended():
                print("game over")
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
              + f"moves with {result} ({state}).")
        translate = {"*": 0, "1-0": 1, "0-1": -1, "1/2-1/2": 0}
        score = translate[result] 

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
        print(f"played {REPORT_CYCLE} moves in {str(elapsed)[0:5]} " \
                + f"seconds, average of {str(avg_per_move)[0:4]} " \
                + "seconds per move.")
        return time_now
        

if __name__ == "__main__":
    SHOW_GAME = True
    RUNS_PER_MOVE = 10
    stale_mate = "8/8/8/8/8/8/R7/5K1k b - - 10 20" 
    match = SelfMatch()
    match.set_game_state(stale_mate)
    match.simulate()


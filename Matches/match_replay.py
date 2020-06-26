'''
Replay matches from pickled move lists.
'''
import os
import argparse
import pickle

from Interpreter import game
from Azts import screen 
from Azts import utility

from Azts.config import WHITE, BLACK


class MatchReplay():
    '''
    Replay matches from pickled move lists.
    This gives a visualisation of game states
    along with the statistical data that is
    being tracked with mlflow.
    '''
    def __init__(self, moves_file, player=None):
        self.game = game.Game()
        self.match_moves = pickle.load(open(moves_file, "rb"))
        self.move_count = 0
        
        self.screen = screen.Screen()
        self.show_game = True
        self._show_game()
        self.player = player

    def _show_game(self):
        img = self.game.render_game()
        self.screen.show_img(img)

    def _one_move_forward(self):
        if self.move_count < len(self.match_moves):
            self.game.make_move(self.match_moves[self.move_count])
            self.move_count += 1 
            print(f"this is move {self.move_count} in " \
                    + f"turn {int((self.move_count + 1)/2)}.")
            print(f"{self.game.board.fen()}")
        else:
            print("this is the final position")

        if self.show_game:
            self._show_game()

    def _jump_to_move(self, move):
        self.game.reset()
        self.move_count = 0
        self.show_game = False
        for i in range(move - 1):
            self._one_move_forward()
        self.show_game = True
        self._one_move_forward()

    def _print_help(self):
        print("Press [enter] to step through game.\n" \
                + "Input a number to jump to respective move.\n" \
                + "Enter \"exit\" to exit the replay.\n" \
                + "Enter \"clear\" to clear the screen.\n" \
                + "Enter \"help\" to display this message.\n")

    def _print_player(self):
        gamestate = self.game.board.fen()
        if gamestate.split(" ")[1] == "w":
            self.player.set_color(WHITE)
        else:
            self.player.set_color(BLACK)

        self.player.reset()
        self.player.set_game_state(gamestate)
        move = self.player.simulate_move()
        print(self.player)
        self.player.reset()


    def replay(self): 
        self._print_help()
        while True:
            user_input = input()
            try:
                user_input = int(user_input)
            except:
                pass 

            if isinstance(user_input, int):
                self._jump_to_move(user_input)
            if user_input == "exit":
                break
            if user_input == "help":
                self._print_help()
            if user_input == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
            if user_input == "tree" and self.player == None:
                print("No player loaded. Load players with option -p.")
            if user_input == "tree" and self.player != None:
                self._print_player() 
            if user_input == "":
                self._one_move_forward()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Player to " \
            + "replay matches from pickled move lists. " \
            + "Use [enter] to go to next move, type in " \
            + "a number to jump to a specific move " \
            + "or press [q] to quit.")
    parser.add_argument("-m", "--match", \
            type=str, \
            help="path to pickled game file.")
    parser.add_argument("-p", "--player", \
            type=str, \
            help="load a player to see his tree search " \
            + "given the current game position with command " \
            + "\"tree\"." )
    parser.add_argument("-r", "--rollouts_per_move",
            type=int, default=100, \
            help="Simulation runs for each move.")

    args = parser.parse_args()

    player = None

    if args.player != None:
        player = utility.load_player(args.player)
        player.set_rollouts_per_move(args.rollouts_per_move)


    replay = MatchReplay(moves_file=args.match, \
            player=player)
    
    replay.replay()




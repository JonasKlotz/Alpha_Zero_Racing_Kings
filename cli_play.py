import argparse
from azts.config import *
from azts import player
from azts import screen
from Interpreter import game
from Interface import TensorNotation

STD_FEN = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"


class CLIPlay:

    def __init__(self, args):
        self.ai_color = WHITE if args.fen.split()[1] == 'w' else BLACK
        self.ai_player = player.Player(self.ai_color, RUNS_PER_MOVE)
        self.game = game.Game()
        if args.show:
            self.screen = screen.Screen()
        self.args = args
        self.set_game_state(args.fen)

    def set_game_state(self, fen_state):
        self.ai_player.set_game_state(fen_state)
        self.game.board.set_board_fen(fen_state)

    def simulate(self):
        while True:
            # check break conditions
            if self.game.is_ended():
                break

            move = ''
            if (self.ai_color==WHITE and self.game.get_current_player()) or (self.ai_color==BLACK and not self.game.get_current_player()):
                # AI player to move
                move = self.ai_player.make_move()
                self.game.make_move()
            else:
                # CLI player to move
                move = self._get_cli_player_move()
                self.game.make_move(move)
                self.ai_player.receive_move(move)

                self._show_game()

        print(f'The game ended with {self.game.board.result()}')



    def _get_cli_player_move(self):
        legal_moves = self.game.get_moves_observation()

        while True:
            player_input = player_input('\n\nPlease enter a move in UCI notation or the new board state in FEN -> for '
                                        'help enter "help" for a list of legal moves enter "list"')
            if 'help' in player_input:
                print('\n\nA move in UCI notation consists of the starting position and the ending position on the '
                      'board. E.g. "a1a2"')
                print('FEN is well described on Wikipedia')
            elif 'list' in player_input:
                print(f'\n\nList of legal moves: {legal_moves}')
            else:
                print('\n\n')
                if player_input in legal_moves:
                    return player_input
                else:
                    print('No legal UCI move detected, checking for FEN')
                    try:
                        move = TensorNotation.move_from_two_fen(self.game.get_observation(), player_input)
                        if move in legal_moves:
                            return move
                        else:
                            raise Exception
                    except:
                        print('No FEN String representing a legal next board position detected')
                    print('Please try again')

    def _show_game(self):
        if args.show:
            img = self.game.render_game()
            self.screen.show_img(img)


if __name__ == "__main__":

    # --- Setting up argparse ---
    parser = argparse.ArgumentParser(description='Start a game against the AlphaZero Racing Kings AI.'
                                     , epilog='Have fun playing! :)'
                                     , fromfile_prefix_chars='@')

    parser.add_argument('-f', '--fen', help='board position in FEN (default: starting position in FEN)'
                        , default=STD_FEN, type=str, dest='fen', metavar='<FEN as String>')

    parser.add_argument('-m', '--getmove', help='only get the next move instead of starting a game (one-shot-mode)'
                        , action='store_true', dest='get_move')

    parser.add_argument('-s', '--show', help='enable game board rendering (graphical representation)'
                        , action='store_true', dest='show')

    parser.add_argument('-p', '--player', help='specify ai engine to play against'
                        , metavar='playername', action='store', type=str, dest='player')

    args = parser.parse_args()

    # ------it status


    print('Im here')

    fen = args.fen
    # TODO: check if FEN is legal
    player_color = WHITE if fen.split()[1] == 'w' else BLACK
    if args.get_move:
        print(f'Calculating next move from position {fen} as {"white" if player_color==WHITE else "black"}')
    else:
        print(f'Starting game from position {fen} as {"white" if player_color==WHITE else "black"}')
        cli_play = CLIPlay(args)
        cli_play.simulate()

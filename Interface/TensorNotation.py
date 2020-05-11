import numpy as np

# which piece to put in which plane when black is to move
piece_indices_black = {
    "k": 0
    , "q": 1
    , "r": 2
    , "b": 3
    , "n": 4
    , "K": 5
    , "Q": 6
    , "R": 7
    , "B": 8
    , "N": 9
}

# which piece to put in which plane when white is to move
piece_indices_white = {
    "K": 0
    , "Q": 1
    , "R": 2
    , "B": 3
    , "N": 4
    , "k": 5
    , "q": 6
    , "r": 7
    , "b": 8
    , "n": 9
}

# start position
std_fen = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"


def fen_to_tensor(fen=std_fen):
    """
    Converts FEN String to tensor notation

    Args:
        fen: Board in FEN notation as String. If no FEN is provided the start position is used

    Returns: np.array representing the board in tensor notation

    """
    tensor = np.zeros((8, 8, 11))
    fen = fen.split()

    # replace digits by number of "1"s and split String into rows
    for i in range(2, 9):
        board_fen = fen[0].replace(str(i), "1" * i).split("/")

    # check who is to move, set indices and tensor accordingly
    if fen[1] == "w":
        piece_indices = piece_indices_white
    else:
        piece_indices = piece_indices_black
        tensor[:8, :8, 10].fill(1)

    # iterate through board and fill tensor
    for i in range(8):
        for j in range(8):
            if not board_fen[i][j].isdigit():
                tensor[i, j, piece_indices[board_fen[i][j]]] = 1

    return tensor

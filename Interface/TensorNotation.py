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

piece_letters_black = [
    "k"
    , "q"
    , "r"
    , "b"
    , "n"
    , "K"
    , "Q"
    , "R"
    , "B"
    , "N"
]

piece_letters_white = [
    "K"
    , "Q"
    , "R"
    , "B"
    , "N"
    , "k"
    , "q"
    , "r"
    , "b"
    , "n"
]

# start position
std_fen = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"


def fen_to_tensor(fen=std_fen):
    """
    Converts FEN String to tensor notation

    Args:
        fen: str representation of board in FEN notation. If no FEN is provided the start position is used

    Returns: np.array representing the board in tensor notation

    """

    tensor = np.zeros((8, 8, 11))
    fen = fen.split()

    # replace digits by number of "1"s and split String into rows
    for i in range(2, 9):
        fen[0] = fen[0].replace(str(i), "1" * i)
    board_fen = fen[0].split("/")
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


def tensor_to_fen(tensor):
    """

    Args:
        tensor: np.array board in tensor notation

    Returns: str representing the board in FEN notation

    """

    fen = ""

    # find color
    if tensor[0, 0, 10] == 0:
        fen_color = "w"
        piece_letters = piece_letters_white
    else:
        fen_color = "b"
        piece_letters = piece_letters_black

    # iterate through board
    for i in range(8):
        count_empty = 0  # count empty tiles
        for j in range(8):
            piece_found = False  # to help count empty tiles
            for k in range(10):
                # piece found
                if tensor[i, j, k]:
                    # add number for empty tiles before
                    if count_empty > 0:
                        fen += str(count_empty)
                        count_empty = 0
                    # add piece
                    fen += piece_letters[k]
                    piece_found = True
            if not piece_found:
                count_empty += 1
        # add remaining empty tiles
        if count_empty > 0:
            fen += str(count_empty)
        # add slash
        if i < 7:
            fen += "/"

    fen += " " + fen_color + " - - 0 1"

    return fen


def get_direction(coord1, coord2):
    """

    Args:
        coord1: np.array with start position
        coord2: np.array with end position

    Returns: int For queen-moves -> 0 for North, counting clockwise to 7 for North-West
                 For knight-moves -> -1 for 2 up 1 right, counting clockwise to -8 for 2 up 1 left

    """

    direction = 0
    diff = coord2 - coord1
    # check for knight-move
    non_zero = np.count_nonzero(np.unique(np.abs(diff)))

    # if knight-move
    if non_zero > 1:
        knight_move_to_indices = [
            [0, -7, 0, -6, 0]
            , [-8, 0, 0, 0, -5]
            , [0, 0, 0, 0, 0]
            , [-1, 0, 0, 0, -4]
            , [0, -2, 0, -3, 0]
        ]
        direction = knight_move_to_indices[diff[0] + 2][diff[1] + 2]

    else:
        queen_move_to_dir = [
            [7, 6, 5]
            , [0, None, 4]
            , [1, 2, 3]
        ]
        diff = np.divide(diff, np.abs(diff).max())

        direction = queen_move_to_dir[int(diff[0]) + 1][int(diff[1]) + 1]

    return direction


def move_to_tensor_indices(uci):
    """

    Args:
        uci: str of UCI representation of move

    Returns: np.array of position in move-tensor as indices

    """

    indices = np.zeros(3)

    # convert to tensor row and column coordinates
    coord1 = np.array([ord(uci[0].upper()) - ord("A"), 8 - int(uci[1])])
    coord2 = np.array([ord(uci[2].upper()) - ord("A"), 8 - int(uci[3])])

    direction = get_direction(coord1, coord2)

    # if knight-move
    if direction < 0:
        indices = [coord1[0], coord1[1], abs(direction) + 55]

    else:
        diff = coord2 - coord1
        indices = [coord1[0], coord1[1], (np.abs(diff).max() - 1) * 8 + direction]

    return indices


def move_to_tensor(uci):
    """

    Args:
        uci: str of UCI representation of move

    Returns: np.array representing the move in tensor notation

    """

    indices = move_to_tensor_indices(uci)

    tensor = np.zeros((8, 8, 64))

    tensor[indices[0], indices[1], indices[2]] = 1

    return tensor


def tensor_indices_to_move(indices):
    """

    Args:
        indices: np.array with indices of move in tensor

    Returns: str of move in UCI format

    """

    # starting tile
    uci = chr(ord("A") + indices[0]).lower() + str(8 - indices[1])

    diff = np.zeros(2)

    # if knights move
    if indices[2] >= 56:
        knight_indices_to_move = [
            [1, -2]
            , [2, -1]
            , [2, 1]
            , [1, 2]
            , [-1, 2]
            , [-2, 1]
            , [-2, -1]
            , [-1, -2]
        ]

        diff = np.array(knight_indices_to_move[indices[2] - 56])

    else:
        direction_to_move = [
            [0, -1]
            , [1, -1]
            , [1, 0]
            , [1, 1]
            , [0, 1]
            , [-1, 1]
            , [-1, 0]
            , [-1, 1]
        ]

        direction = indices[2] % 8
        distance = indices[2] // 8

        diff = np.array(direction_to_move[direction]) * (distance + 1)

    # add ending tile
    uci += chr(ord("A") + indices[0] + int(diff[0])).lower() + str(8 - indices[1] - int(diff[1]))

    return uci


def tensor_to_move(tensor):
    """

    Args:
        tensor: np.array representing the move in tensor notation

    Returns: str of move in UCI format

    """

    # find the max value and give indices to tensor_indices_to_move()
    uci = tensor_indices_to_move(np.array(np.unravel_index(tensor.argmax(), tensor.shape)))

    return uci


def move_from_two_tensors(from_tensor, to_tensor):
    """

    Args:
        from_tensor: np.array representing the board before move in tensor notation
        to_tensor: np.array representing the board after move in tensor notation

    Returns: str of move in UCI format (no checking for legality)

    """
    converted_from_tensor = np.zeros((8, 8, 11))
    if not from_tensor[0, 0, 10] == to_tensor[0, 0, 10]:
        converted_from_tensor[:, :, :5] = from_tensor[:, :, 5:10]
        converted_from_tensor[:, :, 5:10] = from_tensor[:, :, :5]
        converted_from_tensor[:, :, 10] = from_tensor[:, :, 10]
    else:
        converted_from_tensor = from_tensor

    diff = np.subtract(converted_from_tensor[:, :, :10], to_tensor[:, :, :10])
    from_index = np.argwhere(diff == 1)
    to_index = np.argwhere(diff == -1)

    print(diff)

    print(from_index)
    print(to_index)

    uci = chr(ord("A") + from_index[0][1]).lower() + str(8 - from_index[0][0])
    uci += chr(ord("A") + to_index[0][1]).lower() + str(8 - to_index[0][0])

    return uci

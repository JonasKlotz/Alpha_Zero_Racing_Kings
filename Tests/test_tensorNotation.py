# import pytest
from Interface import TensorNotation as tn

MOVES = ['h2h3', 'h2g3', 'g2g8', 'g2g7',\
        'g2g6', 'g2g5', 'g2g4', 'g2g3',\
        'f2a7', 'f2b6', 'f2c5', 'f2h4',\
        'f2d4', 'f2g3', 'f2e3', 'e2f4',\
        'e2d4', 'e2g3', 'e1f3', 'e1d3', 'e1c2']

for i in MOVES:
    tensor = tn.move_to_tensor(i)
    fen = tn.tensor_to_move(tensor)
    if fen != i:
        raise ValueError(f"mismatch: original {i}, back-translated {fen}")

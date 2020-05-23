from azts import self_match
from azts.config import *


num_of_simulations = 25
runs_per_move = 100
match = self_match.SelfMatch(runs_per_move)
outcomes = [0] * NUM_OF_OUTCOMES

for i in range(num_of_simulations):
    outcome = match.simulate()
    outcomes[outcome] += 1
    del match
    match = self_match.SelfMatch(runs_per_move)

print(f"outcomes of {num_of_simulations} games:")
for i, j in enumerate(outcomes):
    if j > 0:
        print(TO_STRING[i] + f": {j} occurences " \
                + f"({str(100 * j / num_of_simulations)[0:5]} %).")


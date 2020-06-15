'''
inspirations to give a name
to your new ai player
'''
import os
import numpy as np 

from Azts.config import RESOURCESDIR

adjectives = []
animals = []

with open(os.path.join(RESOURCESDIR, \
        "animals.txt")) as animal_file:
    animals = animal_file.readlines()

with open(os.path.join(RESOURCESDIR, \
        "adjectives.txt")) as adj_file:
    adjectives = adj_file.readlines()

for i in range(20):
    select = (np.random.rand(2) * np.array([1310, 284])).astype(int)
    print(adjectives[select[0]][:-1] + animals[select[1]][:-1])

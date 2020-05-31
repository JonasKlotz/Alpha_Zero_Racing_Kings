'''
inspirations to give a name
to your new ai player
'''
import numpy as np


adjectives = []
animals = []

with open("animals.txt") as animal_file:
    animals = animal_file.readlines()

with open("adjectives.txt") as adj_file:
    adjectives = adj_file.readlines()

for i in range(20):
    select = (np.random.rand(2) * np.array([1310, 284])).astype(int)
    print(adjectives[select[0]][:-1] + animals[select[1]][:-1])

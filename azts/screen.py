"""
Shows boards from game.
"""

import os.path
import matplotlib.pyplot as plt
from PIL import Image

from azts.config import *


class Screen():
    def __init__(self):
        img_path = os.path.join(RESOURCESDIR, "startboard.png")
        self.img = Image.open(img_path)
        self.graph = plt.imshow(self.img)
        plt.ion()
        plt.show()

    def show_img(self, img):
        self.graph.set_data(img)
        plt.pause(0.01)


if __name__ == "__main__":
    SCREEN = Screen()

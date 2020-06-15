# pylint: disable=E0401
# pylint: disable=E0602
"""
Shows boards from game.
"""

import os.path
import matplotlib.pyplot as plt
from PIL import Image

from Azts.config import RESOURCESDIR


class Screen():
    '''
    module to put up a graphic rendering
    of current game state, updated with
    player moves
    '''
    def __init__(self):
        img_path = os.path.join(RESOURCESDIR, "startboard.png")
        self.img = Image.open(img_path)
        self.graph = plt.imshow(self.img)
        plt.ion()
        plt.show()

    def show_img(self, img):
        '''
        update graphic rendering. needs
        a plt.pause to actually get enough
        cpu priority to do the rendering
        '''
        self.graph.set_data(img)
        plt.pause(0.01)


# pylint: enable=E0401
# pylint: enable=E0602

import matplotlib.pyplot as plt 
from PIL import Image

class Screen():
    def __init__(self):
        self.img = Image.open("startboard.png")
        self.graph = plt.imshow(self.img)
        plt.ion()
        plt.show()

    def show_img(self, img):
        self.graph.set_data(img)
        plt.pause(0.01)


if __name__ == "__main__":
    screen = Screen()
        
        


from sys import path
from os.path import dirname as dir
import pathlib

parentdir = None

if "__file__" in locals():
    # executing from file
    parentdir = pathlib.Path(__file__).parent.absolute()
else:
    parentdir = pathlib.Path().absolute()

while parentdir.as_posix()[-17:] != "AlphaZero-Gruppe1":
    parentdir = parentdir.parent
    if parentdir.as_posix() == "/":
        print("failed to find root dir. Aborting.")
        break


path.append(parentdir.as_posix())

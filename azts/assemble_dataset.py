import os
import pickle
import config


dataset = []


counter = 0
for filename in os.listdir(config.GAMEDIR):
    if "game_" in filename and filename.endswith(".pkl"):
        filename = config.GAMEDIR + "/" + filename
        counter += 1
        game_data = pickle.load(open(filename, "rb"))
        for i in game_data:
            dataset.append(i)
        print(f"added {filename} to dataset.")


dataset_name = f"dataset_{counter}_games.pkl"

pickle.dump(dataset, open(dataset_name, "wb"))
print(f"saved data of {counter} games to {dataset_name}.")

del dataset

test_load = pickle.load(open(dataset_name, "rb"))
print(f"verified integrity of file.\n" \
        + f"dataset is of type {type(test_load)},\n" \
        + f"has {len(test_load)} entries of type {type(test_load[0])}\n" \
        + f"with {len(test_load[0])} entries of type\n" \
        + f"{type(test_load[0][0])}, {type(test_load[0][1])}, " \
        + f"{type(test_load[0][2])}.")

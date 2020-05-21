if __name__ == "__main__":

    from lib.timing import timing

    from Model.model import AZero
    from Model.config import Config

    print("===== Executing Test Run =====")

    # build model
    config = Config('Model/config.yaml')
    azero = AZero(config)

    # load data
    # FILE = '_Data/training_data/dataset_685_games.pkl'
    FILE = 'Model/game_0000.pkl'
    import pickle
    with open(FILE, 'rb') as f:
        train_data = pickle.load(f)

    # wrap training loop for time test
    @timing
    def time_train():
        azero.train(train_data)

    time_train()

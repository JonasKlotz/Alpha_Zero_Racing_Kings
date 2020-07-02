import os
import re
import numpy as np

from mlflow.tracking import MlflowClient


def softmax(Z):
    A = np.exp(Z - Z.max())
    return A / A.sum()


def get_members(obj):
    return [(a, getattr(obj, a)) for a in dir(obj) if "__" not in a]


def valid_tensor(t):
    return np.isfinite(t.numpy()).any()


def valid_ndarray(t):
    return np.isfinite(t).any() and t.any()


def get_latest_dataset_file(_dir):
    """ Returns newest dataset file in game dir """
    files = os.listdir(_dir)
    files = [f for f in files if ".pkl" in f]
    if len(files) == 0:
        return None

    def key_map(file):
        return os.path.getmtime(os.path.join(_dir, file))
    newest_file = max(files, key=key_map)
    return os.path.join(_dir, newest_file)


def prepare_dataset(train_data):
    """ Transforms dataset to format that keras.model expects """
    x_train, y_train_p, y_train_v = np.hsplit(np.array(train_data), [1, 2])
    x_train = np.stack(x_train.flatten(), axis=0)
    y_train_p = np.stack(y_train_p.flatten(), axis=0)
    y_train_v = np.stack(y_train_v.flatten(), axis=0)
    y_train = {"policy_head": y_train_p,
               "value_head": y_train_v}
    return x_train, y_train


def mlflow_get_latest_version(model_name):
    def key_map(ml_model):
        return ml_model.version
    model_list = MlflowClient().get_latest_versions(model_name)
    if len(model_list) == 0:
        return 0
    latest = max(model_list, key=key_map)
    return latest.version


def newest_checkpoint_file(checkpoint_dir):
    """ searches checkpoint dir for newest checkpoint,
    goes by file mtime..
    Returns:
        file (str): filepath
        epoch (int): checkpoint's epoch
    """
    def key_map(file):   # XXX go for newest epoch?
        return os.path.getmtime(os.path.join(checkpoint_dir, file))

    files = reversed(sorted(os.listdir(checkpoint_dir), key=key_map))

    for file_name in files:
        file = os.path.join(checkpoint_dir, file_name)
        if os.path.isfile(file):
            reg = re.search(r"^0*(\d+).*?\.hdf5", file_name)
            if reg is not None:
                epoch = int(reg.group(1))
                return file, epoch
    return None, 0

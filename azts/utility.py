import os.path
import string
import random

from Player import config
from Model.model import AZero
from azts.config import GAMEDIR, PLAYERDIR
from azts import player
from azts import mock_model

GAME = "game"
STATS = "stats"
MOVES = "moves"


# from https://pynative.com/python-generate-random-string/
def random_string(length=8):
    '''
    generate random string as an id stamp
    for self plays
    '''
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def get_player_as_string(player):
    '''
    translates player config objects into
    strings: name.version.modelrevision
    :param config player: player to translate
    :return str: name.version.revision
    '''
    player_str = f"{player.name}.v" \
        + f"{str(player.config_version)}.m" \
        + f"{str(player.model_revision)}"
    return player_str


def get_match_player_names(player1, player2):
    '''
    takes two player config objects as 
    input and generates a string 
    playerX.v.r-playerY.v.r which can be used
    to name files.
    the names are sorted alphabetically and
    thus can be reconstructed 
    '''

    player_names = [i.model_name for i in
                    [player1, player2]]

    player_names.sort()
    match_name = f"{player_names[0]}-{player_names[1]}"

    return match_name


def get_unused_match_handle(player1, player2):
    '''
    create an unused match handle to store all
    data attached to this match
    '''
    if type(player1) is str:
        player1 = load_player_conf(player1)
    if type(player2) is str:
        player2 = load_player_conf(player2)

    match_player = f"{get_match_player_names(player1, player2)}"
    return test_handle(match_player, False)[0]


def test_handle(match_player, isfile):
    '''
    test if a handle is already in use and
    create another one if so
    '''
    handle = f"{match_player}_{random_string()}"
    test = []
    for i in [GAME, MOVES, STATS]:
        test.append(os.path.join(GAMEDIR, f"{i}_{handle}_0000.pkl"))
    for i in test:
        isfile = isfile or os.path.isfile(i)
    while isfile:
        handle, isfile = test_handle(match_player, False)

    return handle, isfile


def get_unused_filepath(name_pattern, folder, i=0):
    '''
    for the given pattern and folder, find the
    next filename that does not overwrite existing
    files
    :return str: full path to unused filename
    '''
    filenumber = i

    filenumberstring = str(filenumber).zfill(4)
    filename = f"{name_pattern}_{filenumberstring}.pkl"
    filepath = os.path.join(folder, filename)
    while os.path.isfile(filepath):
        filenumber += 1
        filenumberstring = str(filenumber).zfill(4)
        filename = f"{name_pattern}_{filenumberstring}.pkl"
        filepath = os.path.join(folder, filename)

    return filepath


def load_player_conf(location):
    '''
    load player configuration from .yaml-path
    '''
    if not "Player/" in location:
        location = "Player/" + location

    if not ".yaml" in location:
        location = location + ".yaml"

    player = config.Config(location)
    return player


def load_model(config, mock=False):
    '''
    load model from configuration
    :param Configuration conf: configuration
    of model
    :param boolean mock: load random generator
    instead
    '''
    if mock:
        model = mock_model.MockModel()
    elif config.stockfish.enabled:
        pass    # TODO: return stockfish model
    else:
        model = AZero(config)

    return model


def load_player(location, mock=False):
    '''
    load player from .yaml-path
    :param str location: relative path 
    to yaml configuration file
    :return player: configured player
    object
    '''
    config = load_player_conf(location)

    model = load_model(config, mock)
    new_player = player.Player(name=config.name,
                               **(config.player.as_dictionary()))

    return new_player


def load_players(loc_1, loc_2, mock=False):
    '''
    load players from .yaml-configuration file
    locations.
    :return list: returns list of two players.
    if the two specified locations are the same,
    self play is assumed and the two players
    share the same model
    '''

    selfplay = loc_1 == loc_2

    locations = [loc_1, loc_2]
    configurations = [load_player_conf(i) for i in locations]

    players = []
    models = []

    if selfplay:
        # same model for both players
        model = load_model(configurations[0], mock)
        models = [model, model]
    else:
        models = [load_model(i, mock) for i in configurations]

    for model, config in zip(models, configurations):
        players.append(player.Player(model=model,
                                     name=config.name,
                                     **(config.player.as_dictionary())))  # ~* dynamite

    return players

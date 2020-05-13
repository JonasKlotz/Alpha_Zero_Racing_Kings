#!/usr/bin/env python3

from model import AZero

azero = AZero('Model/config.yaml')
azero.summary()

# graphviz (not a python package) has to be installed https://www.graphviz.org/
azero.plot_model()

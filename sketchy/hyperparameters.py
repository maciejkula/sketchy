import os

import pickle

import hyperopt
from hyperopt import Trials, fmin


def optimize(objective, space, trials_fname=None, max_evals=5):

    if trials_fname is not None:
        if os.path.exists(trials_fname):
            with open(trials_fname, 'rb') as trials_file:
                trials = pickle.load(trials_file)
        else:
            trials = Trials()

    fmin(objective,
         space=space,
         algo=hyperopt.tpe.suggest,
         trials=trials,
         max_evals=max_evals)

    if trials_fname is not None:
        with open(trials_fname, 'wb') as trials_file:
            pickle.dump(trials, trials_file)

    return trials

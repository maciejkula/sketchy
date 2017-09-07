import argparse

import numpy as np

import torch

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import random_train_test_split

from sketchy.factorization import get_objective, hyperparameter_space
from sketchy.hyperparameters import optimize
from sketchy.sampling import sparsify
from sketchy.results import summarize_trials


CUDA = torch.cuda.is_available()


def load_data(dataset, drop_fraction, random_state):

    dataset = get_movielens_dataset(dataset)
    dataset = sparsify(dataset, drop_fraction, random_state)

    train, rest = random_train_test_split(dataset,
                                          test_percentage=0.2,
                                          random_state=random_state)
    test, validation = random_train_test_split(rest,
                                               test_percentage=0.5,
                                               random_state=random_state)

    return train, validation, test


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run experiment.')
    parser.add_argument('dataset',
                        type=str,
                        help='Name of dataset to use')
    parser.add_argument('--num_trials',
                        default=10,
                        type=int,
                        help='Number of trials to run')
    parser.add_argument('--sparsify',
                        default=1.0,
                        type=float,
                        help='Sparsification fraction')

    args = parser.parse_args()

    random_state = np.random.RandomState(42)

    train, validation, test = load_data(args.dataset,
                                        args.sparsify,
                                        random_state)
    print(train)

    objective = get_objective(train, validation, test)
    space = hyperparameter_space()

    fname = 'factorization_trials_{}_{}.pickle'.format(args.dataset,
                                                       args.sparsify)

    for iteration in range(args.num_trials):
        print('Iteration {}'.format(iteration))
        trials = optimize(objective,
                          space,
                          trials_fname=fname,
                          max_evals=iteration + 1)

        summarize_trials(trials)

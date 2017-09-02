import numpy as np

import torch

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import (random_train_test_split,
                                        user_based_train_test_split)
from spotlight.layers import BloomEmbedding, ScaledEmbedding
from spotlight.factorization.representations import BilinearNet
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.evaluation import mrr_score

from sketchy.layers import LSHEmbedding
from sketchy.factorization import get_objective, hyperparameter_space
from sketchy.hyperparameters import optimize


CUDA = torch.cuda.is_available()


def load_data(random_state):

    dataset = get_movielens_dataset('100K')

    train, rest = random_train_test_split(dataset,
                                          test_percentage=0.2,
                                          random_state=random_state)
    test, validation = random_train_test_split(rest,
                                               test_percentage=0.5,
                                               random_state=random_state)

    return train, validation, test


def build_factorization_model(train, random_state):

    embedding_dim = 64

    item_embeddings = LSHEmbedding(embedding_dim,
                                   num_hash_functions=1)
    item_embeddings.fit(train.tocsr().T)
    user_embeddings = LSHEmbedding(embedding_dim,
                                   num_hash_functions=1)
    user_embeddings.fit(train.tocsr())

    network = BilinearNet(train.num_users,
                          train.num_items,
                          user_embedding_layer=user_embeddings,
                          item_embedding_layer=item_embeddings)

    model = ImplicitFactorizationModel(loss='bpr',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=5*1e-2,
                                       embedding_dim=embedding_dim,
                                       l2=1e-6,
                                       representation=network,
                                       use_cuda=CUDA,
                                       random_state=random_state)

    return model


if __name__ == '__main__':

    random_state = np.random.RandomState(42)

    train, validation, test = load_data(random_state)

    objective = get_objective(train, validation, test)
    space = hyperparameter_space()

    max_evals = 5

    for iteration in range(1, max_evals):
        print('Iteration {}'.format(iteration))
        trials = optimize(objective,
                          space,
                          trials_fname='factorization_trials.pickle',
                          max_evals=iteration)

    # model = build_factorization_model(train, random_state)
    # model.fit(train, verbose=True)

    # mrr = mrr_score(model, test, train=train).mean()

    # print('MRR {}'.format(mrr))

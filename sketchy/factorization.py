import time

from hyperopt import Trials, pyll, hp, fmin, STATUS_OK

import numpy as np

import torch

from spotlight.evaluation import mrr_score
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.representations import BilinearNet
from spotlight.layers import ScaledEmbedding

from sketchy.layers import LSHEmbedding


CUDA = torch.cuda.is_available()


def hyperparameter_space():

    space = {
        'batch_size': hp.quniform('batch_size', 512, 1024, 100),
        'learning_rate': hp.loguniform('learning_rate', -5, -2),
        'l2': hp.loguniform('l2', -10, -4),
        'embedding_dim': hp.quniform('embedding_dim', 16, 128, 10),
        'n_iter': hp.quniform('n_iter', 5, 25, 1),
        'loss': hp.choice('loss', ['adaptive_hinge']),
        'model': hp.choice('lsh', [
            {
                'type': 'lsh',
                'embed': hp.choice('embed', [True, False]),
                'gated': hp.choice('gated', [True, False]),
                'num_hash_functions': hp.quniform('num_hash_functions', 1, 4, 1),
                'residual': hp.choice('residual', [True, False]),
                'num_layers': hp.quniform('num_layers', 1, 3, 1),
                'nonlinearity': hp.choice('nonlinearity', ['tanh', 'relu'])
            },
            {
                'type': 'embedding'
            }
        ])
    }

    return space


def get_objective(train, validation, test):

    random_state = np.random.RandomState(42)

    def objective(hyper):

        print(hyper)

        start = time.clock()

        if hyper['model']['type'] == 'lsh':
            num_hashes = int(hyper['model']['num_hash_functions'])
            num_layers = int(hyper['model']['num_layers'])
            nonlinearity = hyper['model']['nonlinearity']
            residual = hyper['model']['residual']
            embed = hyper['model']['embed']
            gated = hyper['model']['gated']

            item_embeddings = LSHEmbedding(train.num_items,
                                           int(hyper['embedding_dim']),
                                           embed=embed,
                                           gated=gated,
                                           residual_connections=residual,
                                           nonlinearity=nonlinearity,
                                           num_layers=num_layers,
                                           num_hash_functions=num_hashes)
            item_embeddings.fit(train.tocsr().T)
            user_embeddings = LSHEmbedding(train.num_users,
                                           int(hyper['embedding_dim']),
                                           embed=embed,
                                           gated=gated,
                                           residual_connections=residual,
                                           nonlinearity=nonlinearity,
                                           num_layers=num_layers,
                                           num_hash_functions=num_hashes)
            user_embeddings.fit(train.tocsr())
        else:
            user_embeddings = ScaledEmbedding(train.num_users,
                                              int(hyper['embedding_dim']),
                                              padding_idx=0)
            item_embeddings = ScaledEmbedding(train.num_items,
                                              int(hyper['embedding_dim']),
                                              padding_idx=0)

        network = BilinearNet(train.num_users,
                              train.num_items,
                              user_embedding_layer=user_embeddings,
                              item_embedding_layer=item_embeddings)

        model = ImplicitFactorizationModel(loss=hyper['loss'],
                                           n_iter=int(hyper['n_iter']),
                                           batch_size=int(hyper['batch_size']),
                                           learning_rate=hyper['learning_rate'],
                                           embedding_dim=int(hyper['embedding_dim']),
                                           l2=hyper['l2'],
                                           representation=network,
                                           use_cuda=CUDA,
                                           random_state=random_state)

        model.fit(train, verbose=True)

        elapsed = time.clock() - start

        print(model)

        validation_mrr = mrr_score(model, validation, train=train).mean()
        test_mrr = mrr_score(model, test, train=train.tocsr() + validation.tocsr()).mean()

        print('MRR {} {}'.format(validation_mrr, test_mrr))

        return {'loss': -validation_mrr,
                'status': STATUS_OK,
                'validation_mrr': validation_mrr,
                'test_mrr': test_mrr,
                'elapsed': elapsed,
                'hyper': hyper}

    return objective

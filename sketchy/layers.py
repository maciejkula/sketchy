import numpy as np
import scipy.sparse as sp

from sklearn.utils import murmurhash3_32

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from spotlight.layers import ScaledEmbedding


SEEDS = [
    179424941, 179425457, 179425907, 179426369,
    179424977, 179425517, 179425943, 179426407,
    179424989, 179425529, 179425993, 179426447,
    179425003, 179425537, 179426003, 179426453,
    179425019, 179425559, 179426029, 179426491,
    179425027, 179425579, 179426081, 179426549
]


class LSHEmbedding(nn.Module):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 residual_connections=False,
                 embed=False,
                 gated=False,
                 num_layers=1,
                 nonlinearity='tanh',
                 num_hash_functions=4):

        super(LSHEmbedding, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._num_hash_functions = num_hash_functions
        self._masks = SEEDS[:self._num_hash_functions]
        self._num_layers = num_layers
        self._embed = embed
        self._gated = gated

        if nonlinearity == 'tanh':
            self._nonlinearlity = F.tanh
        elif nonlinearity == 'relu':
            self._nonlinearlity = F.relu
        else:
            raise ValueError('Nonlinearity must be one of (tanh, relu)')
        self._residual_connections = residual_connections

        self.layers = [nn.Linear(self._embedding_dim,
                                 self._embedding_dim)
                       for _ in range(self._num_layers)]

        for i, layer in enumerate(self.layers):
            self.add_module('fc_{}'.format(i),
                            layer)

        if embed:
            self._embeddings = ScaledEmbedding(num_embeddings,
                                               embedding_dim)
            if gated:
                self._gate = nn.Linear(self._embedding_dim, 1)

        self._inputs = None

    def fit(self, interactions_matrix):

        def _hash(x, seed):

            result = murmurhash3_32(x, seed)
            return result % self._embedding_dim

        interactions_matrix = interactions_matrix.tocoo()

        hashed_user_ids = np.concatenate(
            [_hash(interactions_matrix.col, seed)
             for seed in self._masks]
        )

        item_ids = np.repeat(interactions_matrix.row, self._num_hash_functions)

        embeddings = sp.coo_matrix((np.ones_like(item_ids),
                                    (item_ids, hashed_user_ids)),
                                   shape=(interactions_matrix.shape[0],
                                          self._embedding_dim),
                                   dtype=np.float32).todense()
        embeddings = np.squeeze(np.asarray(embeddings))
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1), 1.0)[:, np.newaxis]

        self._inputs = Variable(torch.from_numpy(embeddings))

    def forward(self, indices):

        if self._inputs is None:
            raise ValueError('Embeddings have not been fit.')

        original_shape = indices.size()

        if not indices.is_contiguous():
            indices = indices.contiguous()

        if indices.is_cuda and not self._inputs.is_cuda:
            self._inputs = self._inputs.cuda()

        indices = indices.view(-1)

        x = torch.index_select(self._inputs,
                               0,
                               indices.squeeze())

        input_ = x

        for layer in self.layers:
            if self._residual_connections:
                x = x + self._nonlinearlity(layer(x))
            else:
                x = self._nonlinearlity(layer(x))

        if self._embed:
            if self._gated:
                gate = F.sigmoid(self._gate(input_))
                x = (1 - gate) * x + gate * self._embeddings(indices)
            else:
                x = x + self._embeddings(indices)

        return x.view(original_shape + (self._embedding_dim,))

import numpy as np

from spotlight.interactions import Interactions


def sparsify(interactions, drop_fraction, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState()

    indices = random_state.rand(len(interactions)) > drop_fraction

    it = interactions

    return Interactions(it.user_ids[indices],
                        it.item_ids[indices],
                        timestamps=it.timestamps[indices],
                        num_users=it.num_users,
                        num_items=it.num_items)

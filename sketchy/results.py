import numpy as np


def _is_normal(trial):

    return not _is_residual_embedding(trial) and not _is_embedding_free(trial)


def _is_residual_embedding(trial):

    return (trial['result']['hyper']['model']['type'] == 'lsh' and
            trial['result']['hyper']['model']['embed'])


def _is_embedding_free(trial):

    return (trial['result']['hyper']['model']['type'] == 'lsh' and not
            trial['result']['hyper']['model']['embed'])


def _get_best_trial(trials, filter_fnc):

    return sorted([x for x in trials if filter_fnc(x)],
                  key=lambda x: x['result']['loss'])[:1]


def summarize_trials(trials):

    best_normal = _get_best_trial(trials.trials,
                                  _is_normal)
    best_lsh = _get_best_trial(trials.trials,
                               _is_residual_embedding)
    best_embedding_free = _get_best_trial(trials.trials,
                                          _is_embedding_free)

    if best_normal:
        print('Best normal {}'.format(best_normal[0]))

    if best_lsh:
        print('Best LSH embedding {}'.format(best_lsh[0]))

    if best_embedding_free:
        print('Best embedding free {}'.format(best_embedding_free[0]))

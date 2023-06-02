import numpy as np
from .utility import rescale_min_max

def load_training_set(patches_file, part_train=0.8, part_test=0.2, part_val=None, seed=0, reshape=True):
    Y,X = np.load(patches_file)
    for i in range(Y.shape[0]):
        Y[i] = rescale_min_max(Y[i])
        X[i] = rescale_min_max(X[i])
    if part_val:
        x_train, x_val, x_test = split_training_set(X, part_train=part_train,
                                                        part_test=part_test, part_val=part_val,
                                                        seed=seed, reshape=reshape)
        y_train, y_val, y_test = split_training_set(Y, part_train=part_train,
                                                        part_test=part_test, part_val=part_val,
                                                        seed=seed, reshape=reshape)
        return x_train, x_val, x_test, y_train, y_val, y_test
    else:
        x_train, x_test = split_training_set(X, part_train=part_train,
                                                part_test=part_test, part_val=part_val,
                                                seed=seed, reshape=reshape)
        y_train, y_test = split_training_set(Y, part_train=part_train,
                                                part_test=part_test, part_val=part_val,
                                                seed=seed, reshape=reshape)
        return x_train, x_test, y_train, y_test

def split_training_set(total_set, part_train=0.8, part_test=0.2, part_val=None, seed=None, reshape=True):
    ntotal = total_set.shape[0]
    npix = total_set.shape[1]
    indx = np.arange(ntotal)
    ntrain = int(ntotal*part_train)
    ntest = int(ntotal*part_test)
    if seed:
        np.random.seed(seed)
        train_indx = np.random.choice(indx, ntrain)
        indx = np.delete(indx, train_indx)
        test_indx = np.random.choice(indx, ntest)
    else:
        train_indx = indx[0:ntrain]
        test_indx = indx[ntrain:ntrain+ntest]
    train = total_set[train_indx]
    test = total_set[test_indx]
    if part_val:
        if seed:
           val_indx = np.delete(indx, test_index)
        else:
            val_indx = indx[ntrain+ntest:]
        nval = len(val_indx)
        val = total_set[val_indx]
    if reshape:
        train = train.reshape(ntrain, npix, npix, 1)
        test = test.reshape(ntest, npix, npix, 1)
        if part_val:
            val = val.reshape(nval, npix, npix, 1)
    if part_val:
        return train, val, test
    else:
        return train, test

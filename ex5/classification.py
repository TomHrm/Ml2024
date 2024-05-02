# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def load_data(name, m=None):
    data = np.load(name)
    x = data[:,:-1]
    y = data[:,-1]

    return (x, y)

def plot_numbers(numb, tag, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed=66)
    ones = numb[tag == 0]
    fives = numb[tag == 1]

    (fig, axs) = plt.subplots(nrows=2, ncols=4)

    indx = rng.integers(len(ones), size=4)
    for (ax,i) in zip(axs[0], indx):
        ax.imshow(ones[i].reshape(28,28), cmap='gray', vmin=0, vmax=1)

    indx = rng.integers(len(fives), size=4)
    for (ax,i) in zip(axs[1], indx):
        ax.imshow(fives[i].reshape(28,28), cmap='gray', vmin=0, vmax=1)
    
    plt.show()


if __name__ == '__main__':
    x_train, y_train = load_data('dataset_numbers_train.npy')
    plot_numbers(x_train, y_train)

    ...

    print('Train score is: ...')

    x_test, y_test = load_data('dataset_numbers_test.npy')
    print('Test score is: ...')

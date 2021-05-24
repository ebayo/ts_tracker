# Plot metrics, losses and fitness from multiple experiments.

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str,
                        help='file with the names of the experiments to plot')
    parser.add_argument('exp_path', type=str,
                        help='location of the experiments')
    parser.add_argument('-n', '--name', type=str, default='output',
                        help='Path to save the picture')
    parser.add_argument('--metrics', action='store_true')
    parser.add_argument('--losses', action='store_true')
    parser.add_argument('--fitness', action='store_true')

    param = parser.parse_args()

    os.makedirs(param.name, exist_ok=True)

    metrics = {8: 'Precision', 9: 'Recall', 10: 'mAP@.5', 11: 'mAP@.5:.95'} # See initial comments on k_folds_graphs.py
    losses = {12: 'Box loss', 13: 'Object loss', 14: 'Classification loss'}

    columns = []
    just_fitness = False
    if param.metrics:
        columns += list(metrics.keys())
    elif param.fitness:
        columns += [10, 11]
        just_fitness = True
    if param.losses:
        columns += list(losses.keys())

    f_files = open(param.exp, 'r')
    exp_names = f_files.readlines()
    f_files.close()
    exp_names = [f[:-1] for f in exp_names]
    result_files = [os.path.join(param.exp_path, f, 'results.txt') for f in exp_names]

    results = [np.loadtxt(f, usecols=columns) for f in result_files]

    epochs = [len(r) for r in results]
    e_min = min(epochs)
    results = [r[:e_min] for r in results]

    # res_array.shape = (num_exp , e_min, num_metrics(4))
    res_array = np.array(results)

    epochs = np.arange(e_min)
    if param.metrics or param.fitness:
        i0 = 8
    elif param.losses:
        i0 = 12

    if not just_fitness:
        for col in columns:
            i = col-i0
            title = metrics[col] if col in metrics else losses[col]
            ymax = 1 if col in metrics else None
            plt.figure(i, tight_layout=True, figsize=[8, 6])
            plt.plot(epochs, np.transpose(res_array[:, :, i]), linewidth=1)
            plt.title(title)
            plt.xlabel('epochs')
            plt.axis([0, e_min, 0, ymax])
            plt.legend(exp_names)
            plt.grid(True)
            plt.savefig(os.path.join(param.name, title + '.png'))

    if param.fitness:
        fitness = 0.1 * res_array[:, :, 2] + 0.9 * res_array[:, :, 3]
        plt.figure(4, tight_layout=True, figsize=[8, 6])
        plt.plot(epochs, np.transpose(fitness), linewidth=1)
        plt.title('Fitness')
        plt.xlabel('epochs')
        plt.axis([0, e_min, 0, 1])
        plt.legend(exp_names)
        plt.grid(True)
        plt.savefig(os.path.join(param.name, 'Fitness.png'))

    plt.show()



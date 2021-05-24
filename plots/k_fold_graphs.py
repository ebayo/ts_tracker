# Script to show the results of a k-fold validation training

# results.txt headers
# 0 Epoch --> same as index
# 1 gpu_mem
# 2 GIoU --> Generalised Intersection over Union (https://giou.stanford.edu/GIoU.pdf)
# 3 obj --> Objectness
# 4 cls
# 5 total
# 6 targets
# 7 img_size
# 8 P --> Precision
# 9 R --> Recall
# 10 mAP@.5 --> mean Average Precision
# 11 mAP@.5:.95 --> Mean Average Precision
# 12 val_los: box
# 13 val_loss: obj
# 14 val_loss: cls

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

METRICS = {'P': {'col': 8, 'name': 'Precision'},
           'R': {'col': 9, 'name': 'Recall'},
           'map5': {'col': 10, 'name': 'mAP@.5'},
           'map95': {'col': 11, 'name': 'mAP@.5:.95'},
           'fit': {'col': [10, 11], 'name': 'Fitness'}}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str,
                        help='name of the file with the locations of the results files for each fold')
    parser.add_argument('-m', '--metric', choices=[*METRICS.keys(), 'all'], default='all',
                        help='Metric to show the graphic')

    parser.add_argument('-n', '--name', type=str, default='output/results.png',
                        help='Name and path to save the picture')

    param = parser.parse_args()

    f_files = open(param.files, 'r')
    result_files = f_files.readlines()
    f_files.close()
    result_files = [f[:-1] for f in result_files]

    # k = len(result_files)

    if param.metric == 'all':
        metric = [*METRICS.keys()]
        txt_col = [m['col'] for m in METRICS.values()]
    else:
        metric = [param.metric]
        txt_col = [METRICS[param.metric]['col']]

    results = [np.loadtxt(f, usecols=txt_col) for f in result_files]

    # Reduce length of all folds to the minimum length
    epochs = [len(r) for r in results]
    e_min = min(epochs)
    results = [r[:e_min] for r in results]

    # res_array.shape = (k , e_min, len(txt_col))
    res_array = np.array(results)
    # to obtain one measurement for all folds: p = arr[:,:,index_p_in_txt_col]
    means = np.mean(res_array, axis=0)
    mins = np.amin(res_array, axis=0)
    maxs = np.amax(res_array, axis=0)

    if len(metric) > 1:
        n_cols = 2
        n_rows = - (-len(metric) // 2)
        fig1, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex='all')  # squeeze=False??
        epochs = np.arange(e_min)

        for i in range(n_rows):
            for j in range(n_cols):
                axes[i, j].plot(epochs, means[:, i * n_cols + j])  # [:, i * n_cols + j]
                axes[i, j].fill_between(epochs, mins[:, i * n_cols + j], maxs[:, i * n_cols + j], alpha=.5)
    else:
        epochs = np.arange(e_min)
        fig1 = plt.plot(epochs, means)
        plt.fill_between(epochs, mins, maxs, alpha=.5)

    plt.savefig(param.name)
    plt.show()


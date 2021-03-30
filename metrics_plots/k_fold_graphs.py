# Script to show the results of a k-fold validation training

# metrics = ['Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size', 'Class', 'Images' 'Targets',
# 'P', 'R', 'mAP@.5', 'mAP@.5:.95']

# results.txt headers
# 0 Epoch --> same as index
# 1 gpu_mem
# 2 GIoU --> Generalised Intersection over Union (https://giou.stanford.edu/GIoU.pdf)
# 3 obj --> Objectness
# 4 cls
# 5 total
# 6 targets
# 7 img_size
# 8 Class
# 9 Images
# 10 Targets
# 11 P --> Precision
# 12 R --> Recall
# 13 mAP@.5 --> mean Average Precision
# 14 mAP@.5:.95 --> Mean Average Precision

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# TODO:
#  [ ] check name and extension of name / change "default" according to other parameters (files and metrics)
#  [ ] add more metrics
#  [ ] add titles to graphs
#  [ ] see how can we pass more that one metric
#  [ ] check the content of the columns
#  [ ] need of prints??


METRICS = {'P': {'col': 11, 'name': 'Precision'},
           'R': {'col': 12, 'name': 'Recall'},
           'map5': {'col': 13, 'name': 'mAP@.5'},
           'map95': {'col': 14, 'name': 'mAP@.5:.95'}}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', type=str,
                        help='name of the file with the locations of the results files for each fold')
    # parser.add_argument('-c', '-classes', type=str,
    #                    help='File with the name of the classes, same used in database_split')
    parser.add_argument('-m', '--metric', choices=[*METRICS.keys(), 'all'],
                        help='Metric to show the graphic')

    parser.add_argument('-n', '--name', type=str, default='results.png',
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


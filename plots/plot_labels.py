# Script to plot the labels in the dataset and other similar ideas
# plot the labels per class
# plot the number of labels per image

import argparse
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# TODO:
#   [ ] Set pot values so it looks good

def get_label_paths(data_base):
    files = []
    if isinstance(data_base['train'], list):
        files += data_base['train']
    elif data_base['train'] is not None:
        files.append(data_base['train'])
    if isinstance(data_base['val'], list):
        files += data_base['val']
    elif data_base['val'] is not None:
        files.append(data_base['val'])

    return [f.replace('images', 'labels') for f in files]


def get_labels_count(labels_path):
    lbs = []
    lb_image = []

    for f in tqdm(os.listdir(labels_path)):
        if os.path.getsize(os.path.join(labels_path, f)):
            lab = np.loadtxt(os.path.join(labels_path, f), usecols=0, dtype=int, ndmin=1) # read as integers
            # lab is np.ndarray
            lbs += lab.tolist()
            lb_image.append(len(lab))
        else:
            lb_image.append(0)
    return lbs, lb_image


def autolabel(rects, ax):
    # Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base', type=str,
                        help='yaml file passed for training yolov5 with class names and image locations')
    parser.add_argument('--plot_name', type=str, default='plots/histograms.png')

    param = parser.parse_args()

    with open(param.data_base) as file:
        db = yaml.load(file, Loader=yaml.FullLoader)
        print("Read file {} successfully".format(param.data_base))

    label_paths = get_label_paths(db)

    labels = []
    label_count = []

    for lp in label_paths:
        lb, lb_c = get_labels_count(lp)
        labels += lb
        label_count += lb_c

    hist_labels = [labels.count(i) for i in range(db['nc'])]

    # account for background images (0 labels) and 1+ to add the max
    hist_images = [label_count.count(i) for i in range(0, 1 + max(label_count))]

    print('Found {} labels in {} images'.format(len(labels), len(label_count)))

    print('making plots')
    fig = plt.figure()
    ax1, ax2 = fig.subplots(nrows=1, ncols=2)

    # Plot histogram of samples per class
    # ax1.hist_labels(labels, rwidth=0.8)
    width = 0.8
    x = np.arange(db['nc'])
    # TODO: canviar color?? mirar memòria, color=(xx, yy, zz) --> ax1/2.bar(...., color=color)
    rect1 = ax1.bar(x, hist_labels, width, color='#007dcd')
    ax1.set_xlabel('classes')
    ax1.set_xticks(x)
    ax1.set_xticklabels(db['names'], rotation=90)
    ax1.set_title('Samples per class')
    autolabel(rect1, ax1)

    # ax2.hist(label_count, bins=max(label_count))
    x = np.arange(0, 1 + max(label_count))
    rect2 = ax2.bar(x, hist_images, width, color='#007dcd')
    ax2.set_xlabel('Samples per image')
    ax2.set_xticks(x)
    ax2.set_ylabel('Number of images')
    ax2.set_title('Object instances per image')
    autolabel(rect2, ax2)

    plt.savefig(param.plot_name)
    plt.show()

    # top=0.963,
    # bottom=0.155,
    # left=0.028,
    # right=0.992,
    # hspace=0.2,
    # wspace=0.082

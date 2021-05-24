# script that takes a folder with images and their labels (.txt files in YOLO format) and splits the images
# into training and validation set in the structure needed to train YoloV5

# Evolved with the labeling tool (Yolo_mark, CVAT) and the split method (train/val, k-fold)
# k-split-database.py: Newest clean version for CVAT (labels and images in different folders) and k-fold validation

import argparse
import os
import shutil
import random
import yaml
from sklearn.model_selection import KFold

# IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng'] # Image formats accepted by YOLOv5


def change_base_directory(path):
    if os.path.dirname(os.getcwd()) in path:
        return path.replace(os.path.dirname(os.getcwd()), '..')
    return path


def split_sample(opt):
    if opt.same_folder:
        images, labels = files_from_same_folder(opt.src_db)
        opt.src_labels = opt.src_db
        return split_files(images, labels, opt)
    else:
        images, labels, unused = files_from_diff_folders(opt.src_db, opt.src_labels)
        print('Found {} images with {} label files and {} images without labels'.format(len(images), len(labels), len(unused)))

    if opt.k_folds:
        dirs = k_split_files(images, labels, opt)

        if unused:
            unused_dir = os.path.join(opt.dst_db, 'unused')
            os.makedirs(unused_dir, exist_ok=True)
            src_un = [os.path.join(opt.src_db, f) for f in unused]
            dst_un = [os.path.join(unused_dir, f) for f in unused]
            for i in range(len(unused)):
                shutil.copy(src_un[i], dst_un[i])
    else:
        dirs = split_files(images, labels, opt, unused=unused)
    return dirs


# images and labels in different folders
def files_from_diff_folders(src_img, src_labels):
    img_list = os.listdir(src_img)
    print('Found {} images in source directory'.format(len(img_list)))
    lbl_list = os.listdir(src_labels)
    print('Found {} label files in label directory'.format(len(lbl_list)))

    images = []
    labels = []
    unused = []

    for f in lbl_list:
        if os.path.getsize(os.path.join(src_labels, f)):  # the file is not empty
            labels.append(f)
            fn, fext = os.path.splitext(f)
            i = [img for img in img_list if fn in img]
            images.append(i[0])
        else:
            fn, fext = os.path.splitext(f)
            ll = [img for img in img_list if fn in img]
            unused.append(ll[0])
    return images, labels, unused


# images and labels in the same folder,
def files_from_same_folder(src_db):
    f_list = os.listdir(src_db)
    f_list.sort()
    print('Found {} files in source directory'.format(len(f_list)))
    # images will come before the .txt file when sorted alphabetically (images are .jpg or .png)

    for i in range(0, len(f_list), 2):
        if not (f_list[i].endswith('.jpg') or f_list[i].endswith('.png')):
            print('file {} is not a recognized image'.format(f_list[i]))
            break
        if not f_list[i + 1].startswith(f_list[i][0:-4]):
            print('Image {} has no associated label file')
            break

    images = f_list[::2]
    labels = f_list[1::2]

    return images, labels


def k_make_dirs(k, dst_db):
    # p = os.path.join(dst_db, '_{}'.format(k))
    if dst_db.endswith('/'):
        dst_db = dst_db[:-1]
    p = dst_db + '_fold_{}'.format(k+1)
    os.makedirs(p, exist_ok=True)

    p_img = os.path.join(p, 'images')
    os.makedirs(p_img, exist_ok=True)
    p_labels = os.path.join(p, 'labels')
    os.makedirs(p_labels, exist_ok=True)
    print('Making directory {} and its images and labels subdirectories'.format(p))

    return p_img, p_labels


def k_split_files(images, labels, opt):
    kf = KFold(n_splits=opt.k_folds)

    splits = [list(test) for train, test in kf.split(images)]
    dirs = []

    for k in range(opt.k_folds):
        p_img, p_lab = k_make_dirs(k, opt.dst_db)
        dirs.append(p_img)

        src_img = [os.path.join(opt.src_db, images[i]) for i in splits[k]]
        dst_img = [os.path.join(p_img, images[i]) for i in splits[k]]
        src_lab = [os.path.join(opt.src_labels, labels[i]) for i in splits[k]]
        dst_lab = [os.path.join(p_lab, labels[i]) for i in splits[k]]

        print('Fold {} has {} images and {} label files'.format(k+1, len(src_img), len(src_lab)))

        for i in range(len(splits[k])):
            shutil.copy(src_img[i], dst_img[i])
            shutil.copy(src_lab[i], dst_lab[i])

    return dirs


def split_files(images, labels, opt, unused=None):

    N_img = len(images)
    N_train = round(N_img * opt.train)
    N_val = N_img - N_train
    print(
        'Found {} images with labels. Using {} images for training and {} images for validation'.format(N_img, N_train,
                                                                                                        N_val))

    rand_idx = random.sample(range(0, N_img), N_img)
    train_idx = rand_idx[:N_train]
    train_img = [images[i] for i in train_idx]
    train_lab = [labels[i] for i in train_idx]

    val_idx = rand_idx[N_train:]
    val_img = [images[i] for i in val_idx]
    val_lab = [labels[i] for i in val_idx]

    os.makedirs(opt.dst_db, exist_ok=True)

    img_dir = os.path.join(opt.dst_db, 'images')
    os.makedirs(img_dir, exist_ok=True)
    lab_dir = os.path.join(opt.dst_db, 'labels')
    os.makedirs(lab_dir, exist_ok=True)

    dirs = dict(img_train='', img_val='', lab_train='', lab_val='')
    dirs['img_train'] = os.path.join(img_dir, 'train')
    dirs['img_val'] = os.path.join(img_dir, 'validation')
    dirs['lab_train'] = os.path.join(lab_dir, 'train')
    dirs['lab_val'] = os.path.join(lab_dir, 'validation')
    for d in dirs.values():
        if not os.path.exists(d):
            os.mkdir(d)
            print('Making directory {}'.format(d))

    # Copy Train images and labels
    src_img = [os.path.join(opt.src_db, f) for f in train_img]
    dst_img = [os.path.join(dirs['img_train'], f) for f in train_img]
    src_lab = [os.path.join(opt.src_labels, f) for f in train_lab]
    dst_lab = [os.path.join(dirs['lab_train'], f) for f in train_lab]

    for i in range(N_train):
        shutil.copy(src_img[i], dst_img[i])
        shutil.copy(src_lab[i], dst_lab[i])

    # Copy Validation images and labels
    src_img = [os.path.join(opt.src_db, f) for f in val_img]
    dst_img = [os.path.join(dirs['img_val'], f) for f in val_img]
    src_lab = [os.path.join(opt.src_labels, f) for f in val_lab]
    dst_lab = [os.path.join(dirs['lab_val'], f) for f in val_lab]

    for i in range(N_val):
        shutil.copy(src_img[i], dst_img[i])
        shutil.copy(src_lab[i], dst_lab[i])

    if unused:
        unused_dir = os.path.join(opt.dst_db, 'unused')
        if not os.path.exists(unused_dir):
            os.mkdir(unused_dir)
        src_un = [os.path.join(opt.src_db, f) for f in unused]
        dst_un = [os.path.join(unused_dir, f) for f in unused]
        for i in range(len(unused)):
            shutil.copy(src_un[i], dst_un[i])

    return dirs


def generate_yaml(opt, dirs):

    file = open(opt.classes, 'r')
    names = []
    for line in file:
        if line.endswith('\n'):
            names.append(line[:-1])
        else:
            names.append(line)
    file.close()

    if opt.k_folds:
        for k in range(len(dirs)):
            train = []
            for j in range(len(dirs)):
                if k == j:
                    val = dirs[j]
                else:
                    train.append(dirs[j])

            yaml_dict = dict(train=train,
                             val=val,
                             nc=len(names),
                             names=names)

            f_name = opt.yaml_name
            if f_name.endswith('yaml'):
                f_name, ext = os.path.splitext(opt.yaml_name)

            f_name += '_{}.yaml'.format(k+1)
            f_name = os.path.join(opt.dst_db, f_name)
            with open(f_name, 'w') as f:
                data = yaml.dump(yaml_dict, f)

    else:
        yaml_dict = dict(train=dirs['img_train'],
                         val=dirs['img_val'],
                         nc=len(names),
                         names=names)
        f_name = opt.yaml_name

        if not f_name.endswith('.yaml'):
            f_name = f_name + '.yaml'
        f_name = os.path.join(opt.dst_db, f_name)

        with open(f_name, 'w') as f:
            data = yaml.dump(yaml_dict, f)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_db', type=str,
                        help='complete path to database with images and their labels')
    parser.add_argument('dst_db', type=str,
                        help='complete path to final database, if non-existent it will be created')
    parser.add_argument('classes', type=str,
                        help='complete path to file with class names, one per row')
    parser.add_argument('-t', '--train', type=float, default='0.7',
                        help='percentage of training data [0-1]')
    parser.add_argument('-y', '--yaml_name', type=str, default='database.yaml',
                        help='name to save the yaml to configure database for training in dst_db')
    parser.add_argument('--same_folder', action='store_true',
                        help='If the images and labels are in the same folder, depends on labeling tool')
    parser.add_argument('-s', '--src_labels', type=str,
                        help='complete path to labels/.txt files, if not same_folder')
    parser.add_argument('-k', '--k_folds', type=int,
                        help='Number of splits for k-fold validation')

    param = parser.parse_args()

    param.dst_db = change_base_directory(param.dst_db)

    directories = split_sample(param)
    generate_yaml(param, directories)

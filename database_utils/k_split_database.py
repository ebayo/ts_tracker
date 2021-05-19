# Split files in database is k folds
# We don't check image file formats --> but to train with yolov5 they must be in TODO [copiar llista]

import argparse
import os
from sklearn.model_selection import KFold
import shutil
import yaml
from tqdm import tqdm


def lists_of_files(opt):
    img_list = os.listdir(opt.src_img)
    lbl_list = os.listdir(opt.src_lab)

    images = []
    labels = []
    unlabeled = []

    for img in img_list:
        iname, __ = os.path.splitext(img)
        lname = iname + '.txt'

        if lname in lbl_list and os.path.getsize(os.path.join(opt.src_lab, lname)):
            images.append(img)
            labels.append(lname)
        else:
            unlabeled.append(img)
    print('Found {} images with {} labels and {} images without labels'.format(len(images), len(labels), len(unlabeled)))
    return images, labels, unlabeled


def k_split_files(images, labels, opt):
    # If labels is None, assume it's unlabeled images
    if len(images) == 0:
        print('Found no images.')
        return

    kf = KFold(n_splits=opt.k_folds, shuffle=True)
    splits = [list(test) for train, test in kf.split(images)]
    dirs = []

    for k in range(opt.k_folds):
        p_img, p_lab = k_make_dirs(k, opt.dst_db)
        dirs.append(p_img)

        src_img = [os.path.join(opt.src_img, images[i]) for i in splits[k]]
        dst_img = [os.path.join(p_img, images[i]) for i in splits[k]]

        if labels:
            src_lab = [os.path.join(opt.src_lab, labels[i]) for i in splits[k]]
            dst_lab = [os.path.join(p_lab, labels[i]) for i in splits[k]]
        else:
            src_lab = []
            dst_lab = []

        print('Fold {} has {} images and {} label files'.format(k + 1, len(src_img), len(src_lab)))

        for i in tqdm(range(len(splits[k]))):
            shutil.copy(src_img[i], dst_img[i])
            if labels:
                shutil.copy(src_lab[i], dst_lab[i])

    return dirs


def k_make_dirs(k, dst_db):
    p = os.path.join(dst_db, 'fold_{}'.format(k+1))
    os.makedirs(p, exist_ok=True)

    p_img = os.path.join(p, 'images')
    os.makedirs(p_img, exist_ok=True)
    p_labels = os.path.join(p, 'labels')
    os.makedirs(p_labels, exist_ok=True)

    return p_img, p_labels


def create_yaml(opt, dirs):
    file = open(opt.classes, 'r')
    names = []
    for line in file:
        if line.endswith('\n'):
            names.append(line[:-1])
        else:
            names.append(line)
    file.close()

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

        f_name += '_fold_{}.yaml'.format(k + 1)
        f_name = os.path.join(opt.dst_db, f_name)
        with open(f_name, 'w') as f:
            yaml.dump(yaml_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_img', type=str,
                        help='path to images')
    parser.add_argument('src_lab', type=str,
                        help='path to labels')
    parser.add_argument('dst_db', type=str,
                        help='complete path to final database, if it exists it will add the images')
    parser.add_argument('-k', '--k_folds', type=int, default=5,
                        help='Number of splits for k-fold validation')
    parser.add_argument('--no_labels_ok', action='store_true',
                        help='If we want to accept images with no labels')
    parser.add_argument('--yaml', action='store_true',
                        help='flag if we want to create the yaml files')
    parser.add_argument('-y', '--yaml_name', type=str, default='database.yaml',
                        help='name to save the yaml to configure database for training in dst_db')
    parser.add_argument('--classes', type=str,
                        help='complete path to file with class names, one per row')

    param = parser.parse_args()

    # make sure the dst_path is suitable for the database yaml file and yolo
    if os.path.dirname(os.getcwd()) in param.dst_db:
        param.dst_db = param.dst_db.replace(os.path.dirname(os.getcwd()), '..')

    im_paths, lab_paths, unl_paths = lists_of_files(param)

    print('Copying images with labels...')
    im_dirs = k_split_files(im_paths, lab_paths, param)
    if param.no_labels_ok:
        print('Copying images without labels...')
        k_split_files(unl_paths, None, param)

    # copy unlabeled images to review if we are not accepting unlabeled images
    else:
        un_path = os.path.join(param.dst_db, 'unused')
        os.makedirs(un_path, exist_ok=True)
        src_un = [os.path.join(param.src_img, f) for f in unl_paths]
        dst_un = [os.path.join(un_path, f) for f in unl_paths]
        for i in tqdm(range(len(unl_paths))):
            shutil.copy(src_un[i], dst_un[i])

    if param.yaml:
        create_yaml(param, im_dirs)

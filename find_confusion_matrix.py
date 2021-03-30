import argparse
import yaml
import os
import numpy as np
from tqdm import tqdm

import track_utils.yolov5_bridge as yolo
from ts_utils import lxywhn2lxyxy
import metrics_plots.confusion_matrix as cm


# TODO:
#   [ ] check files exist??
#   [ ] check files (database and weights) have the same number of lines


def inference_and_update(val_folder, weights_file, hyp):
    yolo_model = yolo.Yolo5Model(weights_file, hyp)
    image_loader = yolo.ImageLoader(val_folder)

    for lb_path, img in tqdm(image_loader):
        if img is None:
            break

        gt = np.loadtxt(lb_path, ndmin=2)
        # xywhn --> x1y1x2y2
        (h, w, _) = img.shape
        gt = lxywhn2lxyxy(gt, w, h)
        pred = yolo_model.inference(img)  # (x1, y1, x2, y2, conf, class)
        pred = pred.detach().cpu().numpy()
        conf_matrix.process_batch(pred, gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base', type=str,
                        help='text file with the locations of the yaml files passed for training yolov5 with class '
                             'names and image locations or single yaml file')
    parser.add_argument('weights', type=str,
                        help='text file with the locations of the weights for each fold or a single .pt file')
    parser.add_argument('--single_fold', action='store_true')

    parser.add_argument('-c', '--conf_thr', type=float, default=0.25,
                        help='Confidence threshold for confusion matrix [0..1]')
    parser.add_argument('-u', '--iou_thr', type=float, default=0.45,
                        help='IOU threshold fot confusion matrix [0..1]')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Device to use in YOLOv5 inference')
    parser.add_argument('-s', '--save_dir', type=str, default='data/',
                        help='Directory where to save the confusion matrix as a figure')

    param = parser.parse_args()

    os.makedirs(param.save_dir, exist_ok=True)

    if param.single_fold:
        file = open(param.data_base)
        db = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
        print("Read file {} successfully".format(param.data_base))
        val_folders = [db['val']]
        classes = db['names']
        weights_files = [param.weights]

    else:
        file = open(param.data_base)
        val_folders = []
        for l in file.readlines():
            f = open(l[:-1])
            db = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
            val_folders.append(db['val'])
        file.close()
        classes = db['names']
        # val_folders = [v[:-1] for v in val_folders]

        file = open(param.weights)
        weights_files = file.readlines()
        file.close()
        weights_files = [w[:-1] for w in weights_files]

    # initialize confusion matrix
    # conf_matrix = yolo.ConfusionMatrix(len(classes), param.conf_thr, param.iou_thr)
    conf_matrix = cm.ConfusionMatrix(len(classes), param.conf_thr, param.iou_thr)
    # hyperparameters needed for inference (values from yolov5/test.py)
    hyper = dict(nms_th=0.001, iou_th=0.6, device=param.device)

    for val, weights in zip(val_folders, weights_files):
        print('Processing fold {}/{}'.format(val[-1], len[val_folders]))
        inference_and_update(val, weights, hyper)

    # plot matrix
    # conf_matrix.plot(param.save_dir, classes)
    print(conf_matrix.matrix)
    conf_matrix.plot(param.save_dir, classes)
    save_txt = os.path.join(param.save_dir, 'confusion_matrix.txt')
    np.savetxt(save_txt, conf_matrix.matrix)

import argparse
import yaml
import os
import numpy as np
from tqdm import tqdm
import cv2

import ts_utils.yolov5_bridge as yolo
from ts_utils.bbox_utils import lxywhn2lxyxy
import ts_utils.confusion_matrix as cm


def find_images_with_errors(val_folder, weights_file, hyp, save_folder, matrix, im_size):
    yolo_model = yolo.Yolo5Model(weights_file, hyp, image_size=im_size)
    image_loader = yolo.ImageLoader(val_folder)
    names = yolo_model.names

    for lb_path, img in tqdm(image_loader):
        if img is None:
            break

        if os.path.exists(lb_path):
            gt = np.loadtxt(lb_path, ndmin=2)
        else:
            gt = np.array([[conf_matrix.nc, 0, 0, 0, 0]])

        # xywhn --> x1y1x2y2
        (h, w, _) = img.shape
        gt = lxywhn2lxyxy(gt, w, h)
        pred = yolo_model.inference(img)  # (x1, y1, x2, y2, conf, class)
        pred = pred.detach().cpu().numpy()


        if matrix:
            err = conf_matrix.process_batch_error(pred, gt)
        else:
            err = conf_matrix.detection_error(pred, gt)
        if err:
            color = (0, 255, 0)
            for bbox in gt:
                if bbox[0] != conf_matrix.nc:
                    class_name = names[int(bbox[0])]
                    cv2.rectangle(img, (int(bbox[1]), int(bbox[2])), (int(bbox[3]), int(bbox[4])), color, 2)
                    cv2.rectangle(img, (int(bbox[1]), int(bbox[2] - 30)),
                                  (int(bbox[1]) + (len(class_name)) * 17, int(bbox[2])), color, -1)
                    cv2.putText(img, class_name, (int(bbox[1]), int(bbox[2] - 10)), 0, 0.75,
                                (255, 255, 255), 2)

            color = (0, 0, 255)
            for bbox in pred:
                class_name = names[int(bbox[5])]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
                              (int(bbox[0]) + (len(class_name) + len(str(bbox[4]))) * 17, int(bbox[1])), color, -1)
                cv2.putText(img, class_name + "-" + str(bbox[4]), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                            (255, 255, 255), 2)

            img_name, __ = os.path.splitext(os.path.basename(lb_path))
            img_name += '.png'
            img_path = os.path.join(save_folder, img_name)
            cv2.imwrite(img_path, img)


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
    parser.add_argument('--matrix', action='store_true',
                        help="If we want to find the confusion matrix besides the images with errors")
    parser.add_argument('--im_size', type=int, default=640,
                        help='image size used for training yolo')

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
    conf_matrix = cm.ConfusionMatrix(len(classes), param.conf_thr, param.iou_thr)
    # hyperparameters needed for inference (values from yolov5/test.py)
    # hyper = dict(conf_th=0.001, iou_th=0.6, device=param.device)
    # hyperparameters needed for inference (values from yolov5/detect.py)
    hyper = dict(conf_th=param.conf_thr*0.75, iou_th=param.iou_thr*0.75, device=param.device)

    fold = 1
    for val, weights in zip(val_folders, weights_files):
        print('Processing fold {}/{}'.format(fold, len(val_folders)))
        save = os.path.join(param.save_dir, str(fold))
        os.makedirs(save, exist_ok=True)
        find_images_with_errors(val, weights, hyper, save, param.matrix, param.im_size)
        fold += 1

    # plot matrix
    if param.matrix:
        print(conf_matrix.matrix)
        conf_matrix.plot(param.save_dir, classes)
        save_txt = os.path.join(param.save_dir, 'confusion_matrix.txt')
        np.savetxt(save_txt, conf_matrix.matrix, fmt='%1d')

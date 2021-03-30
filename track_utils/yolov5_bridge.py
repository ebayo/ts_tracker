import torch
import sys
import numpy as np

import glob
import os
import cv2
from pathlib import Path

sys.path.append('../yolov5')
import utils.general as gen
import utils.torch_utils as tu
import utils.datasets as ds
import utils.metrics as met

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


class Yolo5Model:
    def __init__(self, weights, hyp, image_size=640):
        # from yolov5/models/experimental.py>attempt_load()
        self.device = tu.select_device(hyp['device'])
        self.img_size = image_size
        self.nms_th = hyp['nms_th']
        self.iou_th = hyp['iou_th']
        self.model = torch.load(weights, map_location=self.device)['model'].float().fuse().eval()
        print('Model loaded successfully.\n')
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def get_name(self, idx):
        return self.names[idx]

    def inference(self, im0):
        # from yolov5/utils/datasets.py > LoadImages
        # Padded resize
        img = ds.letterbox(im0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        # img = img.half() if half else img.float()     # uint8 to fp16/32
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)[0]
        # print(predictions.shape)

        # Apply NMS with low threshold to do a first discard of predictions
        # predictions is list of boxes (x1, y1, x2, y2, conf, class) --> float

        pred = gen.non_max_suppression(pred, self.nms_th, self.iou_th)
        # print(pred[0].shape)    #--> first dimension is number of classes
        pred = pred[0]  # We only predict on one image

        # Rescale boxes from img_size to im0 size
        if len(pred):
            pred[:, :4] = gen.scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()

        return pred


class ImageLoader:  # for inference
    # Adapted from yolov5/utils/datasets.py

    def __init__(self, path, img_size=640, stride=32):

        # p = str(Path(path).absolute())  # os-agnostic absolute path
        p = path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img = self.cap.read()

            self.frame += 1
            # print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img = cv2.imread(path)  # BGR
            assert img is not None, 'Image Not Found ' + path
            # print(f'image {self.count}/{self.nf} {path}', end='\n')

        lb_path = ds.img2label_paths([path])
        return lb_path[0], img

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

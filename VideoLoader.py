import cv2
import numpy as np

import sys
sys.path.append('../yolov5_ts_detect')
import utils.datasets as ds


class VideoLoader:

    def __init__(self, vid_path):
        self.capture = cv2.VideoCapture(vid_path)
        self.current_frame = 0
        self.num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.img_size = 640

    # CAL??
    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return self.num_frames

    def __next__(self):
        ret_val, img0 = self.capture.read()
        if not ret_val:
            self.capture.release()
        self.current_frame += 1

        # TODO: check Yolo needs this
        # from yolov5/utils/datasets.py > LoadImages
        # Padded resize
        img = ds.letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0

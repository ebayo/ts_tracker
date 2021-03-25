import cv2
import numpy as np
import sys

sys.path.append('../yolov5')
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
            return None, None
        self.current_frame += 1

        # TODO: check Yolo needs this
        # from yolov5/utils/datasets.py > LoadImages
        # Padded resize
        img = ds.letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0

    def close(self):
        self.capture.release()

    def initialise_video_writer(self, file_name):
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_name, codec, fps, (width, height))
        return out

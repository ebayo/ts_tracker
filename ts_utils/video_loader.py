# Video loader for YOLOv5 inference

import cv2


class VideoLoader:

    def __init__(self, vid_path, im_size=0):
        self.capture = cv2.VideoCapture(vid_path)
        self.current_frame = 0
        self.num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.img_size = im_size

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return self.num_frames

    def __next__(self):
        ret_val, img = self.capture.read()
        if not ret_val:
            self.capture.release()
            return None
        self.current_frame += 1
        return img

    def close(self):
        self.capture.release()

    def get_size(self):
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def initialise_video_writer(self, file_name):
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_name, codec, fps, (width, height))
        return out

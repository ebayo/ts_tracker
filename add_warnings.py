# Add pedestrian and car exit warning to a video with detections and tracking
# Detections are in the same folder and have the same name as the video
# Detections are .txt in the format: [frame_idx track_id label x_center y_center b_width b_height] (see track.py)
# Videos in mp4 format only

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from enum import Enum

from ts_utils.video_loader import VideoLoader

names = ['h_bike_ok', 'h_bike_w', 'h_give_way_ok', 'h_checkered', 'h_30', 'h_no_park', 'v_bike_circle', 'v_bike_square']


class WarnType(Enum):
    CAR = {'idx': [5], 'color':(0, 255, 255)}
    PEDESTRIAN = {'idx': [2, 3], 'color': (255, 0, 0)}

    def color(self):
        return self.value['color']

    def id(self):
        return self.value['idx']

    @staticmethod
    def get_type(idx):
        for e in WarnType:
            if idx in e.id():
                return e
        return None

    @staticmethod
    def is_warning(idx):
        for e in WarnType:
            if idx in e.id():
                return True
        return False


class Warn:
    min_life = 25
    active_warnings = {}

    def __init__(self, ts):
        self.frame_init = ts[0]
        self.track_id = ts[1]
        self.label = ts[2]
        self.type = WarnType.get_type(self.label)
        self.life = 0
        self.active = True

    def update(self):
        self.life += 1
        if self.life > Warn.min_life and self.active is False:
            return False
            #Warn.active_warnings.pop(self.track_id)
        return True

'''
    @staticmethod
    def reset():
        Warn.active_warnings.clear()

    @staticmethod
    def update_warnings(detected_ts):

        for ts in detected_ts:
            w = Warn.active_warnings.get(ts[1], None)
            if w is None and ts[2] in WarnType.all_ids():
                w = Warn(ts)
            w.active = True

        for w in Warn.active_warnings:
            w.update()
            w.active = False

        Warn.active_warnings = dict(sorted(Warn.active_warnings.items(), key=lambda item: item[1].life, reverse=True))


def print_warnings(frame):
    numw = 0
    xcoord = len(WarnType.PEDESTRIAN.name + ' - 100') + 4
    for w in Warn.active_warnings:
        cv2.rectangle(frame, (0, numw * 14), (xcoord, (numw + 1) * 14), w.type.color())
        cv2.putText(frame, w.type.name + ' - ' + w.track_id, (2, numw * 14 + 2), 0, 0.75, (255, 255, 255), 2)
        numw += 1

    return frame
'''


def add_warnings(vid_path):
    v_loader = VideoLoader(vid_path)

    width, height = v_loader.get_size()
    xcoord = int(width * 0.22)
    ycoord = int(height * 0.025)

    v_out = os.path.join(param.output_folder, os.path.basename(vid_path))
    v_writer = v_loader.initialise_video_writer(v_out)

    # Detections and tracks
    det_f = vid_path.replace('.mp4', '.txt')
    det = np.loadtxt(det_f, usecols=[0, 1, 2], dtype=int)  # frame_idx, track_id, label
    # frames = det[:, 0]

    active_warnings = {}

    def update_warnings(detected_ts, warnings):
        if len(detected_ts) > 0:
            for ts in detected_ts:
                w = warnings.get(ts[1], None)
                if w is None and WarnType.is_warning(ts[2]):
                    w = Warn(ts)
                    warnings[w.track_id] = w
                if w:
                    w.active = True
        w_del = []
        for w in warnings.values():
            if not w.update():
                w_del.append(w.track_id)
            w.active = False

        for t_id in w_del:
            warnings.pop(t_id)

        return dict(sorted(warnings.items(), key=lambda item: item[1].life, reverse=True))

    frame_id = 0
    for frame in tqdm(v_loader):
        if frame is None:
            print('Video is finished')
            break

        found = np.where(det[:, 0] == frame_id)[0] # indices for found objcts in current frame
        # print(found)
        update_warnings(det[found], active_warnings)

        # if frame contains a "warning" --> add new
        numw = 0

        out = frame
        for warn in active_warnings.values():
            out = cv2.rectangle(frame, (0, numw * ycoord), (xcoord, (numw + 1) * ycoord), warn.type.color(), -1)
            out = cv2.rectangle(out, (0, numw * ycoord), (xcoord, (numw + 1) * ycoord), (255, 255, 255), 3)
            out = cv2.putText(out, warn.type.name + ' - ' + str(warn.track_id), (10, numw * ycoord + 25), 0, 0.75, (255, 255, 255), 2)
            numw += 1

        v_writer.write(out)
        frame_id += 1

    v_loader.close()
    v_writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str,
                        help='Video file or txt with video files')
    parser.add_argument('output_folder', type=str, default='output/videos')

    param = parser.parse_args()

    if param.video.endswith('.txt'):
        path = os.path.dirname(param.video)
        with open(param.video) as f:
            vid_src = f.readlines()
            vid_src = [os.path.join(path, s[:-1]) for s in vid_src]
    else:
        vid_src = [param.video]

    os.makedirs(param.output_folder, exist_ok=True)

    i = 0
    for vid in vid_src:
        print('Processing video {}/{}'.format(i, len(vid_src)))
        add_warnings(vid)
        i += 1
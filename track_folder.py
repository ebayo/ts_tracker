# from ..deep_sort.deep_sort import nn_matching
import argparse
import os

import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ts_utils import deep_sort_bridge as ds
from ts_utils.video_loader import VideoLoader
from ts_utils import yolov5_bridge as yolo
import ts_utils as ts



def track_ts(vid_path, hyp, yolo_net):
    # initialize video loader, yolov5 network and deep_sort tracking objects
    vid_loader = VideoLoader(vid_path)
    # yolo_net = yolo.Yolo5Model(opt.weights, hyp)
    deep_sort = ds.DeepSort(param.descriptor_net, hyp)
    res = vid_path.replace('.mp4', '.txt')
    results_writer = open(res, 'w')

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    vid_out = vid_loader.initialise_video_writer(opt.output_video)

    frame_idx = 0
    for frame in vid_loader:
        print('Processing img {}/{}'.format(frame_idx, vid_loader.num_frames))

        if frame is None:
            print('Video is finished')
            break;

        yolo_detect = yolo_net.inference(frame)  # Columns are [x1 y1 x2 y2 conf class] (not normalized)

        confidences = [b[4] for b in yolo_detect]
        labels = [b[5] for b in yolo_detect]

        if hyp['device'] == 'cpu':
            bboxes = ts.xyxy2tlwh(yolo_detect)
        else:
            bboxes = ts.xyxy2tlwh(yolo_detect).detach().cpu()
        bboxes = bboxes.numpy()

        deep_sort.track(frame, bboxes, confidences, labels)

        # Visualization and save video
        for track in deep_sort.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = yolo_net.get_name(track.label)
            color = colors[int(track.track_id) % len(colors)]
            color = [j * 255 for j in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)

        if opt.display:
            cv2.imshow('output', frame)

        vid_out.write(frame)

        # Save results [img track_id label x1 y1 x2 y2]
        for track in deep_sort.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            results_writer.write('{} {} {} {} {} {} {}\n'.format(frame_idx, track.track_id, track.label,
                                                               bbox[0], bbox[1], bbox[2], bbox[3]))

        frame_idx += 1

    vid_loader.close()
    vid_out.release()
    results_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_folder', type=str,
                        help='video file to analyse')
    parser.add_argument('output_folder', type=str, default='results/videos')
    parser.add_argument('weights', type=str,
                        help='Yolov5 weights')
    parser.add_argument('--hyp', type=str, default='data/hyp_gpu.yaml',
                        help='hyperparameters path')
    parser.add_argument('--descriptor_net', type=str, default='data/mars-small128.pb',
                        help='network used to extract descriptors for the detected bounding boxes')
    parser.add_argument('--result_file', action='store_true',
                        help='whether to save the detected boxes with their class and track')
    parser.add_argument('--display', action='store_true')

    param = parser.parse_args()

    with open(param.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyperparameters

    for k in hyp.keys():
        if isinstance(hyp[k], str) and hyp[k] == 'None':
            hyp[k] = None

    os.makedirs(param.output_folder, exist_ok=True)

    net = yolo.Yolo5Model(param.weights, hyp)

    for vid in tqdm(os.listdir(param.video_folder)):
        if vid.endswith('.mp4'):
            # track
            track_ts(param, hyp, net)

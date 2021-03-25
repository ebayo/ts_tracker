# from ..deep_sort.deep_sort import nn_matching
import argparse
import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np


from video_loader import VideoLoader
import yolov5_bridge as yolo
import deep_sort_bridge as ds
import ts_utils as ts


def track_ts(opt, hyp):
    # initialize video loader, yolov5 network and deep_sort tracking objects
    vid_loader = VideoLoader(opt.video)
    yolo_net = yolo.Yolo5Model(opt.weights, hyp)
    deep_sort = ds.DeepSort(opt.net, hyp, yolo_net.get_names())
    results_writer = open(opt.result_file, 'w')

    vid_out = None
    colors = None

    if opt.save_video or opt.display:
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    if opt.save_video:
        vid_out = vid_loader.initialise_video_writer(opt.output_video)

    frame_idx = 0
    for frame, frame0 in vid_loader:
        print('Processing frame {}/{}'.format(frame_idx, vid_loader.num_frames))

        yolo_detect = yolo_net.inference(frame, frame0)  # Columns are [x1 y1 x2 y2 conf class] (not normalized)

        confidences = [b[4] for b in yolo_detect]
        labels = [b[5] for b in yolo_detect]
        bboxes = ts.xyxy2xywh(yolo_detect)

        deep_sort.track(frame0, bboxes, confidences, labels)

        # Visualization and save video
        if opt.display or opt.save_video:
            for track in deep_sort.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class_name()
                color = colors[int(track.track_id) % len(colors)]
                color = [j * 255 for j in color]
                cv2.rectangle(frame0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame0, (int(bbox[0]), int(bbox[1] - 30)),
                              (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
                cv2.putText(frame0, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                            (255, 255, 255), 2)

        if opt.display:
            cv2.imshow('output', frame0)

        if opt.save_video:
            vid_out.write(frame0)

        # Save results [frame track_id label x1 y1 x2 y2]
        for track in deep_sort.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class_name()
            results_writer.write(str(frame_idx) + ' ' + track.track_id + ' ' + class_name + ' ' +
                               bbox[0] + ' ' + bbox[1] + ' ' + bbox[2] + ' ' + bbox[3] + '\n')

        frame_idx += 1
        if frame_idx > 5:
            break

    vid_loader.close()
    vid_out.release()
    results_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str,
                        help='video file to analyse')
    parser.add_argument('weights', type=str,
                        help='Yolov5 weights')
    parser.add_argument('--hyp', type=str, default='data/hyp_gpu.yaml',
                        help='hyperparameters path')
    parser.add_argument('--descriptor_net', type=str, default='data/mars-small128.pb',
                        help='network used to extract descriptors for the detected bounding boxes')
    parser.add_argument('--result_file', type=str, default='data/results.txt',
                        help='File to save the detected boxes with their class and track')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--output_video', type=str, default='data/output.mp4')
    parser.add_argument('--save_video', action='store_true')

    param = parser.parse_args()

    with open(param.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyperparameters

    for k in hyp.keys():
        if isinstance(hyp[k], str) and hyp[k] == 'None':
            hyp[k] = None

    track_ts(param, hyp)

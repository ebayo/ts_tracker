# from ..deep_sort.deep_sort import nn_matching
import argparse
import yaml

from VideoLoader import VideoLoader
from yolov5_model import YOLOv5

def track_ts(opt, hyp):
    vid_loader = VideoLoader(opt.video)
    yolo_net = YOLOv5(opt.weights, hyp)

    i = 0
    for frame in vid_loader:
        pred = yolo_net.inference(frame)
        print('Prediction after nms'.format(pred))
        i += 1
        if i > 5:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str,
                        help='video file to analyse')
    parser.add_argument('weights', type=str,
                        help='Yolov5 weights')
    parser.add_argument('--hyp', type=str, default='data/hyp.yaml',
                        help='hyperparameters path')

    param = parser.parse_args()

    with open(param.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyperparameters

    track_ts(param, hyp)

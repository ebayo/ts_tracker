# ts_tracker
Use trained YOLOv5 with DeepSORT to track bike traffic signals

Track bike traffic signs using YOLOv5 trained on custom database and DeepSORT for training

track.py is the main script, the others serve as bridges between the pertinent networks and our task

## Requirements

- Yolov5 v5.0 (https://github.com/ultralytics/yolov5) needs to be in the parent directory (or change the location in tolov5_bridge.py)
Use ```git clone --branch v5.0 https://github.com/ultralytics/yolov5.git``` to clone the specific version



- deep_sort (https://github.com/nwojke/deep_sort) needs to be in the parent directory (or change the location in deep_sort_bridge.py)
     - Tensorflow in version 1.5

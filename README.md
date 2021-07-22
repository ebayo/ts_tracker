# ts_tracker

Tools used in the completion of the master thesis "Traffic Sign Detection for Micromobility" for ETSETB, UPC.

Main functionality is tracking unsing a trained YOLOv5 and DeepSORT to track bike traffic signals.

## Tacker

### Requirements

- Yolov5 v5.0 (https://github.com/ultralytics/yolov5) needs to be in the parent directory (or change the location in tolov5_bridge.py)
Use ```git clone --branch v5.0 https://github.com/ultralytics/yolov5.git``` to clone the specific version



- deep_sort (https://github.com/nwojke/deep_sort) needs to be in the parent directory (or change the location in deep_sort_bridge.py)
     - Tensorflow in version 1.5
     - !!! Atenció versió sklearn

### Usage

``` command```

## Other functionalities

The usage of the other scripts is descrived in [detailed_README.md](detailed_README.md)

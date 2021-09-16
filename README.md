# ts_tracker

Tools used in the completion of the master thesis "Traffic Sign Detection for Micromobility" for ETSETB, UPC.

Main functionality is tracking unsing a trained YOLOv5 and DeepSORT to track bike traffic signals.

## Tacker

### Requirements

- [Yolov5 v5.0](https://github.com/ultralytics/yolov5) needs to be in the parent directory (or change the location in yolov5_bridge.py)
Use ```git clone --branch v5.0 https://github.com/ultralytics/yolov5.git``` to clone the specific version used.



- [deep_sort](https://github.com/nwojke/deep_sort) needs to be in the parent directory (or change the location in deep_sort_bridge.py)
     - Tensorflow in version 1.5 (gives a security alert to upgrade to version >=2.4.0)
     - scikit-learn==0.22.2,because it used a depecrated function in later releases. Possible solution [here](https://stackoverflow.com/questions/62390517/no-module-named-sklearn-utils-linear-assignment)

### Usage

Basic usage with default values:

``` python track.py video YOLOv5_weights```

Other options are:

- ```video``` path to video file. Only tested with .mp4
- ```weights``` path to the .pt file with the YOLOv5 trained weights
- ```--hyp data/hyp.gpu.yaml```* hyperparameters path. File YAML with the device (gpu or cpu) and thresholds for both detection and tracking
- ```--descriptor_net data/mars-small128.pb```* network used to extract descriptors for the detected bounding boxes
- ```--result_file path/results.txt```* file to save the detected boxes with their class and track id
- ```--display```* display the detection and tracking on screen, not tested
- ```--output_video path/output.mp4``` path to video file with the tracking boxes
- ```--save_video``` if we want to save the video
- ```--torch``` if we want to use the pytorch implementation of deep_sort, not implemented

To track all .mp4 files in a folder with the same parameters

``` python track_folder.py input/folder path/output path/YOLOv5_weights ```

\* options than can be used when tracking on a folder.

### Output

If we selected the option to save the video, a new video with the result will be created. The results will show as bounding boxes with the corresponding label and an id number respresenting the track. The color of the bounding box relates to the track id not the label itself.

The bounding boxes are also saved in the ```result_file``` and are saved in columns as: ```[frame track_id track_label x1 y1 x2 y2]``` with one row per detection. Boxes saved by the top left and bottom right points 


## Other functionalities

The other scripts in this project are used to plot and change the database used and to study the results from both detection and tracking. A more detailed description and usage of these can be und in [detailed_README.md](detailed_README.md)

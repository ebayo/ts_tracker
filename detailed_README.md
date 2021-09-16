# Detailed description and usage of scripts in the directory


## add_warnings.py

Script used to create a mock-up verison of the tracking results adding a visual warning on the top left corner when certain signs or markigs are detected that warn the cyclist of potenntial car or pedestrian crossings. 

It uses an MP4 video and a text file with the same name and in the same folder containing the detections (the output from [tracking](./track.py))


## confusion_matrix_images.py

Uses [Confusion Matrix for Object Detection](https://github.com/kaanakan/object_detection_confusion_matrix) as a base to obtain a confusion matrix and to take a look at the images where object detections is failing.

### Usage

Basic usage is
```pyton confusion_matrix_images.py data_base weights [options]```

It can be used on a single validation partition, or with k-fold validation by adding up each fold's confussion matrix.


Options are: 
- ```data_base```: yaml file used for YOLOv5 training or a txt file with the locations of the yaml files used in k-fold validation, one per row
- ```weights```: weight file from YOLOv5 (.pt) or txt file with the locations of the weights for each fold
- ```--single_fold```: mark that we are using a single fold, default will be taken as k-fold validation
- ```--conf_thr```: Confidence threshold for a detection to be taken as valid for the confusion matrix
- ```--iou_thr```: IOU threshold between ground truth and detection for the confusion matrix
- ```--device```: device used for YOLOv5 inference
- ```--save_dir```: main output folder to save the matrix as an image and the images with errors, if selected
- ```--im_size```: image size used for training YOLOv5
- ```--save_images```: save images with errors inside save_dir/<fold_num or 1>

### Output
Confusion matrix saved as image in ```save_dir``` or ```output``` with the values normalised and printed in the console with absolute values, i.e. the number of images.

If selected, the images wich contribute wo errors with the bounding boxes of the ground truth in green and their label and the detections in red, their label and confidence score. This helps to see the images where the network is making mistakes more subjectively.

## Data

This folder contains additional data used by the tracker, specifically some hyperparameters and the feature extractor network.

## Database_utils

Contains scripts to manipulate the database.

### change_classes.py

Allows to remove classes and change the indexes of current classes. The inputs are the annotation files and two txt files with one class per row one for the current annotations and one for the new.

### database_split.py

Allows to generate different splits of the database, either test and validation or k-folds and generate the necessary yaml file to train YOLOv5.



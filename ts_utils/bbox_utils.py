# Auxiliary functions to change bounding boxes annotation formats

import numpy as np
import torch


def xyxy2tlwh(x):
    # from yolov5 [x1 y1 x2 y2] --> [x1 y1 w h]
    #y = np.ndarray((x.shape[0], 4))
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]  # x top left
    y[:, 1] = x[:, 1]  # y top left
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y[:, :4]


def lxywhn2lxyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [cl, x, y, w, h] normalized to [cl, x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw  # top left x
    y[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh  # top left y
    y[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw  # bottom right x
    y[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh  # bottom right y
    return y
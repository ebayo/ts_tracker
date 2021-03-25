import numpy as np
import torch

# Auxiliar functions

def xyxy2tlwh(x):
    # from yolov5 [x1 y1 x2 y2] --> [x1 y1 w h]
    #y = np.ndarray((x.shape[0], 4))
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]  # x top left
    y[:, 1] = x[:, 1]  # y top left
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y[:, :4]

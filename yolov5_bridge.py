import torch
import sys

sys.path.append('../yolov5')
import utils.general as gen
import utils.torch_utils as tu


class Yolo5Model:
    def __init__(self, weights, hyp, image_size=640):
        # from yolov5/models/experimental.py>attempt_load()
        self.device = tu.select_device(hyp['device'])
        self.imgsz = image_size
        self.nms_th = hyp['nms_th']
        self.iou_th = hyp['iou_th']
        self.model = torch.load(weights, map_location=self.device)['model'].float().fuse().eval()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def get_name(self, idx):
        return self.names[idx]

    def inference(self, img, im0):
        img = torch.from_numpy(img).to(self.device)
        # img = img.half() if half else img.float()     # uint8 to fp16/32
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)[0]
        # print(predictions.shape)

        # Apply NMS with low threshold to do a first discard of predictions
        # predictions is list of boxes (x1, y1, x2, y, cong, class) --> float
        pred = gen.non_max_suppression(pred, self.nms_th, self.iou_th)
        #print(pred[0].shape)    #--> first dimension is number of classes
        pred = pred[0]  # We only predict on one image

        # Rescale boxes from img_size to im0 size
        if len(pred):
            pred[:, :4] = gen.scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()

        return pred

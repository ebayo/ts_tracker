import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys

sys.path.append('../deep_sort')
import tools.generate_detections as gen
import deep_sort.nn_matching as nn
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.track import Track
from application_util import Visualization, create_unique_color_uchar, NoVisualization


class DeepSort:

    def __init__(self, net, hyp, class_names):
        self.encoder = gen.create_box_encoder(net, batch_size=1)
        self.metric = nn.NearestNeighborDistanceMetric("cosine", hyp['max_cosine_distance'], hyp['nn_budget'])
        self.tracker = self.LbTracker(self.metric)
        self.min_conf = hyp['min_conf']
        self.class_names = class_names
        self.info = None

    def track(self, img, bboxes, confidences, labels):
        features = self.encoder(img, bboxes)
        lb_detections = self.get_detections(bboxes, confidences, features, labels)
        self.tracker.predict()
        self.tracker.update([d.detection for d in lb_detections])

    def get_detections(self, bboxes, confidences, features, labels):
        det = []
        for b, c, f, l in zip(bboxes, confidences, features, labels):
            if c >= self.min_conf:
                det.append(self.LbDetection(b, c, f, int(l)))
        return det

    def tracks_to_string(self):
        s = ''
        for track in self.tracker.tracks:
            s += str(track)

    def init_visualizer(self, im0, video_name, num_frames, display):
        self.info = {'image_size': im0.shape,
                     'sequence_name': video_name,
                     'min_frame_idx': '0',
                     'max_frame_idx': num_frames}

        if display:
            return LbVisualization(self.info, update_ms=5)
        else:
            return NoVisualization

    # Extending some of deep_sort classes to account for multiclass labeling

    class LbDetection(Detection):
        def __init__(self, box, conf, feature, label):
            super().__init__(self, box, conf, feature)
            self.label = label

    class LbTracker(Tracker):
        def __init__(self, metric):
            super().__init__(metric)

        def _initiate_track(self, detection):
            mean, covariance = self.kf.initiate(detection.to_xyah())
            self.tracks.append(self.LbTrack(
                mean, covariance, self._next_id, self.n_init, self.max_age,
                detection.feature, label=detection.label))
            self._next_id += 1

    class LbTrack(Track):
        def __init__(self, label, **kwds):
            self.label = label
            super().__init__(**kwds)

        def get_class_name(self):
            return self.class_names[self.label]


class LbVisualization(Visualization):
    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            label = str(track.label) + ' ' + str(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=label)

# Bridge between Deep_SORT code and the traffic sign tracker.
# Add label information to tracks. Doesn't use it.

import sys

sys.path.append('../deep_sort')
import tools.generate_detections as gen
import deep_sort.nn_matching as nn
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.track import Track, TrackState


class DeepSort:

    def __init__(self, net, hyp):
        self.encoder = gen.create_box_encoder(net, batch_size=1)
        self.metric = nn.NearestNeighborDistanceMetric("cosine", hyp['max_cosine_distance'], hyp['nn_budget'])
        self.tracker = LbTracker(self.metric)
        self.min_conf = hyp['min_conf']
        self.info = None

    def track(self, img, bboxes, confidences, labels):
        features = self.encoder(img, bboxes)
        lb_detections = self.get_detections(bboxes, confidences, features, labels)
        self.tracker.predict()
        self.tracker.update(lb_detections)

    def get_detections(self, bboxes, confidences, features, labels):
        det = []
        for b, c, f, l in zip(bboxes, confidences, features, labels):
            if c >= self.min_conf:
                det.append(LbDetection(b, c, f, int(l)))
        return det


# Extending some of deep_sort classes to account for multiclass labeling

class LbDetection(Detection):
    def __init__(self, box, conf, feature, label):
        Detection.__init__(self, box, conf, feature)
        self.label = label


class LbTracker(Tracker):
    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(LbTrack(mean, covariance, self._next_id, self.n_init, self.max_age,
                                   detection.feature, label=detection.label))
        self._next_id += 1


class LbTrack(Track):
    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, label=''):
        self.label = label
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

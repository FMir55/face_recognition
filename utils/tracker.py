import numpy as np
from norfair import Tracker


class Detection:
  def __init__(self, points: np.array, scores=None):
    # xmin, ymin, xmax, ymax
    self.points = points
    self.scores = scores
    self.label = None

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

def get_tracker(initialization_delay, max_distance_between_points):
  return Tracker(
    distance_function=euclidean_distance,
    distance_threshold=max_distance_between_points,
    initialization_delay = initialization_delay
  )

def convert_detection(obj, scale_x, scale_y):
    bbox = obj.bbox.scale(scale_x, scale_y)
    bbox_out = np.array(
        [
            [int(bbox.xmin), int(bbox.ymin)],
            [int(bbox.xmax), int(bbox.ymax)]
        ]
    )
    score = obj.score
    return Detection(points=bbox_out, scores=score)

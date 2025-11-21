"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import numpy as np
import numpy.typing as npt

from filterpy.kalman import KalmanFilter
import lap

np.random.seed(0)


def linear_assignment(cost_matrix: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
  _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
  return np.array([[y[i],i] for i in x if i >= 0]) #


def iou_batch(bb_test: npt.NDArray[np.floating], bb_gt: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w: float = bbox[2] - bbox[0]
  h: float = bbox[3] - bbox[1]
  x: float = bbox[0] + w/2.
  y: float = bbox[1] + h/2.
  s: float = w * h    #scale is just area
  r: float = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count: int = 0
  def __init__(self, bbox: npt.NDArray[np.floating]) -> None:
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf: KalmanFilter = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update: int = 0
    self.id: int = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history: list[npt.NDArray[np.floating]] = []
    self.hits: int = 0
    self.hit_streak: int = 0
    self.age: int = 0

  def update(self, bbox: npt.NDArray[np.floating]) -> None:
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self) -> npt.NDArray[np.floating]:
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self) -> npt.NDArray[np.floating]:
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(
    detections: npt.NDArray[np.floating], 
    trackers: npt.NDArray[np.floating], 
    iou_threshold: float = 0.3
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer]]:
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix: npt.NDArray[np.floating] = iou_batch(detections, trackers)

  matched_indices: npt.NDArray[np.integer]
  if min(iou_matrix.shape) > 0:
    a: npt.NDArray[np.integer] = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2), dtype=int)

  unmatched_detections: list[int] = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers: list[int] = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches: list[npt.NDArray[np.integer]] = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches_array: npt.NDArray[np.integer] = np.empty((0,2),dtype=int)
  else:
    matches_array = np.concatenate(matches,axis=0)

  return matches_array, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
    """
    Sets key parameters for SORT
    """
    self.max_age: int = max_age
    self.min_hits: int = min_hits
    self.iou_threshold: float = iou_threshold
    self.trackers: list[KalmanBoxTracker] = []
    self.frame_count: int = 0

  def update(self, dets: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks: npt.NDArray[np.floating] = np.zeros((len(self.trackers), 5))
    to_del: list[int] = []
    ret: list[npt.NDArray[np.floating]] = []
    for t, trk_row in enumerate(trks):
      pos: npt.NDArray[np.floating] = self.trackers[t].predict()[0]
      trk_row[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched: npt.NDArray[np.integer]
    unmatched_dets: npt.NDArray[np.integer]
    unmatched_trks: npt.NDArray[np.integer]
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk: KalmanBoxTracker = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i: int = len(self.trackers)
    for trk in reversed(self.trackers):
        d: npt.NDArray[np.floating] = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

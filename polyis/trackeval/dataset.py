import json

import numpy as np

from modules.TrackEval.trackeval.datasets._base_dataset import _BaseDataset


class Dataset(_BaseDataset):
    """Dataset class for Polyis bounding box tracking"""

    @staticmethod
    def get_default_dataset_config():
        return {}

    def __init__(self, config: "dict | None" = None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        config = {} if config is None else config
        self.tracker_list = config.get('tracker_list', ['sort'])
        self.seq_list = config.get('seq_list', [''])
        self.class_list = ['vehicle']

        self.output_fol = config.get('output_fol', 'output-eval')
        self.output_sub_fol = config.get('output_sub_fol', None)
        self.input_gt = config['input_gt']
        self.input_track = config['input_track']
        self.skip = config['skip']
        self.tracker = config['tracker']

    def get_display_name(self, tracker):
        return tracker

    def _load_raw_file(self, tracker, seq: str, is_gt: bool):
        """Load a file (gt or tracker) in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        data = {}
        if is_gt:
            file = self.input_gt
            with open(file, 'r') as f:
                lines = f.readlines()
            
            data['num_timesteps'] = len(lines)
            data['gt_ids'] = []
            data['gt_classes'] = []
            data['gt_dets'] = []
            data['gt_crowd_ignore_regions'] = []
            data['gt_extras'] = []

            for idx, line in enumerate(lines):
                t = json.loads(line)
                frame_idx, dets = t
                if frame_idx % self.skip != 0:
                    idx += 1
                    continue
                assert frame_idx == idx, (frame_idx, idx)
                gt_ids = []
                gt_classes = []
                gt_dets = []
                gt_extras = []
                for det in dets:
                    gt_ids.append(det[0])
                    gt_classes.append(0)
                    gt_dets.append(det[1:])
                    gt_extras.append({})
                data['gt_ids'].append(np.array(gt_ids, dtype=int))
                data['gt_classes'].append(np.array(gt_classes, dtype=int))
                data['gt_dets'].append(np.array(gt_dets))
                data['gt_crowd_ignore_regions'].append([])
                data['gt_extras'].append(gt_extras)
            
            data['seq'] = seq
            return data
        else:
            file = self.input_track
            with open(file, 'r') as f:
                trajectories = f.readlines()
            data['tracker_ids'] = []
            data['tracker_classes'] = []
            data['tracker_dets'] = []
            data['tracker_confidences'] = []

            idx = 0
            for l in trajectories:
                try:
                    t = json.loads(l)
                    frame_idx, tracks = t

                    assert idx == int(frame_idx) / self.skip, (idx, frame_idx, self.skip, seq, self.input_gt)

                    t = np.array(tracks, dtype=float)
                    if len(t) == 0:
                        t = np.empty((0, 5))
                    n, dim = t.shape

                    data['tracker_ids'].append(t[:, 0].astype(int))
                    data['tracker_dets'].append(t[:, 1:5])

                    tracker_classes = np.zeros((n,), dtype=int)
                    if dim > 5:
                        tracker_classes = t[:, 5].astype(int)
                    data['tracker_classes'].append(tracker_classes)
                    
                    tracker_confidences = np.ones((n,), dtype=float)
                    if dim > 6:
                        tracker_confidences = t[:, 6]
                    data['tracker_confidences'].append(tracker_confidences)

                    idx += 1
                    data['num_timesteps'] = idx
                except Exception as e:
                    if len(l) != 0:
                        raise e
            
            data['seq'] = seq
            return data

    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        MOT Challenge:
            In MOT Challenge, the 4 preproc steps are as follow:
                1) There is only one class (pedestrian) to be evaluated, but all other classes are used for preproc.
                2) Predictions are matched against all gt boxes (regardless of class), those matching with distractor
                    objects are removed.
                3) There is no crowd ignore regions.
                4) All gt dets except pedestrian are removed, also removes pedestrian gt dets marked with zero_marked.
        """
        # Check that input data has unique ids
        self._check_unique_ids(raw_data)

        num_tracker_ids = set()
        for ids in raw_data['tracker_ids']:
            num_tracker_ids.update(ids)

        num_gt_ids = set()
        for ids in raw_data['gt_ids']:
            num_gt_ids.update(ids)

        data = {
            'num_timesteps': raw_data['num_timesteps'],

            'gt_ids': raw_data['gt_ids'],
            'gt_dets': raw_data['gt_dets'],
            'gt_classes': raw_data['gt_classes'],
            'gt_crowd_ignore_regions': raw_data['gt_crowd_ignore_regions'],
            'gt_extras': raw_data['gt_extras'],

            'tracker_ids': raw_data['tracker_ids'],
            'tracker_confidences': raw_data['tracker_confidences'],
            'tracker_dets': raw_data['tracker_dets'],
            'tracker_classes': raw_data['tracker_classes'],

            'num_tracker_dets': sum(len(x) for x in raw_data['tracker_ids']),
            'num_gt_dets': sum(len(x) for x in raw_data['gt_ids']),
            'num_tracker_ids': max(num_tracker_ids) + 1 if len(num_tracker_ids) > 0 else 0,
            'num_gt_ids': max(num_gt_ids) + 1,

            'similarity_scores': raw_data['similarity_scores'],
        }

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        return self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='x0y0x1y1')

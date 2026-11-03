import json
import os

import numpy as np

from modules.TrackEval.trackeval.datasets._base_dataset import _BaseDataset


def pad_scoring_window_from_env() -> bool:
    """Padding is the canonical scoring; POLYIS_PAD_SCORING_WINDOW=0 restores TrackEval's truncated window."""
    value = os.environ.get('POLYIS_PAD_SCORING_WINDOW', '1')
    if value not in ('0', '1'):
        raise ValueError(f'POLYIS_PAD_SCORING_WINDOW must be 0 or 1, not {value!r}')
    return value == '1'


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
        self.seq_list = config['seq_list']
        self.class_list = ['vehicle']

        self.output_fol = config.get('output_fol', 'output-eval')
        self.output_sub_fol = config.get('output_sub_fol', None)
        self.input_dir = config['input_dir']
        self.input_gt_dir = config.get('input_gt_dir', self.input_dir)
        self.input_track_dir = config.get('input_track_dir', self.input_dir)
        self.input_gt = config['input_gt']
        self.input_track = config['input_track']
        self.skip = config['skip']
        self.tracker = config['tracker']
        # Score each sequence over the longer of the two files (see get_raw_seq_data); on unless disabled.
        self.pad_scoring_window = config.get('pad_scoring_window', pad_scoring_window_from_env())

    def get_display_name(self, tracker):
        return tracker

    def get_raw_seq_data(self, tracker, seq):
        """TrackEval's raw data, optionally padded so neither file's tail is dropped.

        TrackEval takes the frame count from the ground-truth file and pairs frames with zip, so frames after
        the shorter file's last row are never scored: ground truth after a system's last output row is not
        counted as missed, and output after the ground truth's last row is not counted as a false positive.
        With pad_scoring_window, both sides are loaded over the longer file's length, the shorter one padded
        with empty frames; frames past both files are empty on both sides and do not affect the metrics.
        """
        if not self.pad_scoring_window:
            return super().get_raw_seq_data(tracker, seq)
        # Load both sides, then reload the shorter one over the longer one's length.
        raw_gt = self._load_raw_file(tracker, seq, is_gt=True)
        raw_tracker = self._load_raw_file(tracker, seq, is_gt=False)
        timesteps = max(raw_gt['num_timesteps'], raw_tracker['num_timesteps'])
        if raw_gt['num_timesteps'] < timesteps:
            raw_gt = self._load_raw_file(tracker, seq, is_gt=True, num_timesteps=timesteps)
        if raw_tracker['num_timesteps'] < timesteps:
            raw_tracker = self._load_raw_file(tracker, seq, is_gt=False, num_timesteps=timesteps)
        # Merge as TrackEval does, ground truth last, so num_timesteps is the padded length.
        raw_data = {**raw_tracker, **raw_gt}
        raw_data['similarity_scores'] = [self._calculate_similarities(g, t)
                                         for g, t in zip(raw_data['gt_dets'], raw_data['tracker_dets'])]
        return raw_data

    def _load_raw_file(self, tracker, seq: str, is_gt: bool, num_timesteps: int | None = None):
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
        prefix = 'gt_' if is_gt else 'tracker_'
        input_dir = self.input_gt_dir if is_gt else self.input_track_dir
        file = self.input_gt if is_gt else self.input_track

        with open(os.path.join(input_dir, seq, file), 'r') as f:
            lines = f.readlines()

        raw: dict[int, list[tuple[int, float, float, float, float]]] = {}
        data['num_timesteps'] = 0
        for line in lines:
            if len(line) == 0:
                continue
            t = json.loads(line)
            frame_idx = t['frame_idx']
            dets = t['tracks']
            assert frame_idx not in raw, (frame_idx, raw)
            raw[frame_idx] = dets
            data['num_timesteps'] = max(data['num_timesteps'], frame_idx + 1)
        # Extend the sequence with empty frames when a longer scoring window is requested.
        if num_timesteps is not None:
            data['num_timesteps'] = max(data['num_timesteps'], num_timesteps)
        
        data[f'{prefix}ids'] = []
        data[f'{prefix}classes'] = []
        data[f'{prefix}confidences'] = []
        data[f'{prefix}dets'] = []
        data[f'{prefix}crowd_ignore_regions'] = []
        data[f'{prefix}extras'] = []

        for idx in range(data['num_timesteps']):
            dets = raw.get(idx, [])
            if idx % self.skip != 0:
                continue

            n = len(dets)
            data[f'{prefix}crowd_ignore_regions'].append([])
            data[f'{prefix}extras'].append([{} for _ in range(n)])
            if n == 0:
                data[f'{prefix}ids'].append(np.empty(0, dtype=int))
                data[f'{prefix}dets'].append(np.empty((0, 4), dtype=float))
                data[f'{prefix}classes'].append(np.empty(0, dtype=int))
                data[f'{prefix}confidences'].append(np.empty(0, dtype=float))
                continue
            tfloat = np.array(dets, dtype=float)
            tint = np.array(dets, dtype=int)
            _, dim = tint.shape
            data[f'{prefix}ids'].append(tint[:, 0])
            data[f'{prefix}dets'].append(tfloat[:, 1:5])
            data[f'{prefix}classes'].append(tint[:, 5] if dim > 5 else np.zeros(n, dtype=int))
            data[f'{prefix}confidences'].append(tint[:, 6] if dim > 6 else np.ones(n, dtype=float))
        
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
            'num_gt_ids': max(num_gt_ids) + 1 if len(num_gt_ids) > 0 else 0,

            'similarity_scores': raw_data['similarity_scores'],
        }

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        return self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='x0y0x1y1')

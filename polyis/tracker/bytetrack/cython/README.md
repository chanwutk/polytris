# ByteTrack Cython Implementation

This directory contains a high-performance Cython implementation of the ByteTrack multi-object tracking algorithm.

## Performance

The Cython implementation achieves approximately **5.9x speedup** over the pure Python implementation:
- **Cython**: ~4,658 frames/second (0.21 ms/frame)
- **Python**: ~792 frames/second (1.25 ms/frame)

## Files

- `kalman_filter.pyx/pxd`: 8-dimensional Kalman filter implementation for track prediction
- `matching.pyx`: IOU computation and linear assignment (Hungarian algorithm via LAPJV)
- `bytetrack.pyx`: Main tracker implementation with track management
- `.gitignore`: Excludes generated C/C++ files and compiled extensions

## Implementation Details

### Kalman Filter
- **State vector**: 8D (x, y, aspect_ratio, height, vx, vy, va, vh)
- **Measurement vector**: 4D (x, y, aspect_ratio, height)
- Uses constant velocity motion model

### Matching
- **IOU calculation**: Uses PASCAL VOC formula (adds +1 to dimensions) to match original implementation
- **Linear assignment**: LAPJV algorithm for optimal track-detection association
- **Fuse score**: Combines IOU similarity with detection confidence scores

### Track Management
- **States**: New, Tracked, Lost, Removed
- **Track activation**: Requires consistent detection before activation
- **Track recovery**: Re-activates lost tracks when matched with detections
- **Track removal**: Removes tracks after buffer period without matches

## Building

Build the Cython extensions from the project root:

```bash
docker exec polyis python setup.py build_ext --inplace
```

## Testing

Run the comparison test to verify correctness and measure performance:

```bash
docker exec polyis python -m pytest tests/test_bytetrack_comparison.py -v
```

## Key Implementation Differences from Python

1. **Native Cython structures**: Uses C structs (`STrack`) with functions instead of Python classes with methods
2. **No GIL**: Critical sections use `nogil` for true parallelism potential
3. **IOU formula**: Matches Python's PASCAL VOC formula (+1 to dimensions) for consistency
4. **Numerical precision**: Results match within 1e-5 pixels (0.00001) due to floating-point operations

## Usage

```python
from polyis.tracker.bytetrack.cython.bytetrack import BYTETracker

# Create tracker
class Args:
    track_thresh = 0.5      # Detection threshold for tracking
    track_buffer = 30       # Number of frames to keep lost tracks
    match_thresh = 0.8      # IOU threshold for matching
    mot20 = False           # Use MOT20 evaluation protocol

tracker = BYTETracker(Args(), frame_rate=30)

# Update with detections
detections = np.array([[x1, y1, x2, y2, score], ...], dtype=np.float64)
img_info = (height, width)
img_size = (height, width)

tracks = tracker.update(detections, img_info, img_size)
# Returns: np.array([[x1, y1, x2, y2, track_id], ...])
```

## Notes

- The tolerance for numerical comparison with the Python implementation is set to 1e-5 pixels
- Global track ID counter is managed internally
- Use `reset_tracker_count()` to reset the counter between tracking sessions

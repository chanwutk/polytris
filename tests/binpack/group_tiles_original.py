from queue import Queue
import numpy as np


def find_connected_tiles(bitmap: np.ndarray, i: int, j: int, original_bitmap: np.ndarray, mode: int) -> list[tuple[int, int]]:
    """
    Find all connected tiles in the bitmap starting from the tile at (i, j).
    
    Args:
        bitmap: 2D numpy array representing the grid of tiles,
                where 1 indicates a tile with detection and 0 indicates no detection
        i: row index of the starting tile
        j: column index of the starting tile
        
    Returns:
        list[tuple[int, int]]: List of tuples representing the coordinates of all connected tiles
    """
    value = bitmap[i, j]
    # q = Queue()
    q = []
    q.append((i, j))
    filled: list[tuple[int, int]] = []
    while len(q) > 0:
        i, j = q.pop()
        bitmap[i, j] = value
        filled.append((i, j))
        for _i, _j in [(-1, 0), (0, -1), (+1, 0), (0, +1)]:
            _i += i
            _j += j
            if bitmap[_i, _j] != 0 and bitmap[_i, _j] != value:
                if mode == 0 or mode == 2 or (mode == 1 and (original_bitmap[i, j] == 1 or original_bitmap[_i, _j] == 1)):
                    q.append((_i, _j))
    return filled


def _add_padding(bitmap: np.ndarray) -> np.ndarray:
    pad = np.zeros_like(bitmap)
    pad[:-1, :] += bitmap[1:, :]
    pad[1:, :] += bitmap[:-1, :]
    pad[:, :-1] += bitmap[:, 1:]
    pad[:, 1:] += bitmap[:, :-1]

    pad = np.where(pad, 2, 0)
    pad = np.where(bitmap, 1, pad)
    return pad


res = _add_padding(np.array([[0, 1, 0],
                           [0, 0, 0],
                           [1, 0, 1]]))
exp = np.array([[2, 1, 2],
              [2, 2, 2],
              [1, 2, 1]])
assert np.array_equal(res, exp), f"\n{res}\n{exp}"

res = _add_padding(np.array([[0, 0, 0],
                           [1, 0, 0],
                           [1, 0, 0]]))
exp = np.array([[2, 0, 0],
              [1, 2, 0],
              [1, 2, 0]])
assert np.array_equal(res, exp), f"\n{res}\n{exp}"


def group_tiles(bitmap: np.ndarray, mode: int = 0) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """
    Original Python implementation of group_tiles (backup).
    """
    if mode != 0:
        bitmap = _add_padding(bitmap)

    h, w = bitmap.shape
    _groups = np.arange(h * w, dtype=np.int16) + 1
    _groups = _groups.reshape(bitmap.shape)
    _groups = _groups * (bitmap != 0)
    
    # Padding with size=1 on all sides
    groups = np.zeros((h + 2, w + 2), dtype=np.int16)
    groups[1:h+1, 1:w+1] = _groups
    
    visited: set[int] = set()
    bins: list[tuple[np.ndarray, tuple[int, int]]] = []
    
    padded_bitmap = np.zeros((h + 2, w + 2), dtype=bitmap.dtype)
    padded_bitmap[1:h+1, 1:w+1] = bitmap
    for i in range(groups.shape[0]):
        for j in range(groups.shape[1]):
            if groups[i, j] == 0 or groups[i, j] in visited:
                continue

            connected_tiles = find_connected_tiles(groups, i, j, padded_bitmap, mode)
            if not connected_tiles:
                continue
                
            connected_tiles = np.array(connected_tiles, dtype=int).T
            mask = np.zeros((h + 1, w + 1), dtype=np.uint8)
            mask[*connected_tiles] = True
            
            offset = np.min(connected_tiles, axis=1)
            end = np.max(connected_tiles, axis=1) + 1
            
            mask = mask[offset[0]:end[0], offset[1]:end[1]]
            bins.append((mask, (int(offset[0] - 1), int(offset[1] - 1))))
            visited.add(groups[i, j])
    
    return bins
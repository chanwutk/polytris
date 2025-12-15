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
        original_bitmap: Original bitmap before padding was applied (1=occupied, 2=padding, 0=empty)
        mode: Padding mode (0-6)

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
        # Check all 4 orthogonal neighbors
        for _i, _j in [(-1, 0), (0, -1), (+1, 0), (0, +1)]:
            _i += i
            _j += j
            # If neighbor is non-zero and unvisited
            if bitmap[_i, _j] != 0 and bitmap[_i, _j] != value:
                # Mode 0: Always connect any non-zero neighbors
                # Mode 1-6: Only connect if current cell or neighbor is an original tile (value 1)
                #           This prevents padding tiles (value 2) from connecting to other padding tiles
                if mode == 0 or original_bitmap[i, j] == 1 or original_bitmap[_i, _j] == 1:
                    q.append((_i, _j))
    return filled


def _add_padding(bitmap: np.ndarray, mode: int) -> np.ndarray:
    """
    Add padding to the bitmap based on the specified mode.

    Args:
        bitmap: 2D numpy array representing the grid of tiles
        mode: Padding mode (0-6)
              0 = No padding
              1 = Plus padding (all 4 orthogonal neighbors)
              2 = Top-left padding (top, left, and top-left corner)
              3 = Top-right padding (top, right, and top-right corner)
              4 = Bottom-left padding (bottom, left, and bottom-left corner)
              5 = Bottom-right padding (bottom, right, and bottom-right corner)
              6 = Square padding (all 8 neighbors including diagonals)

    Returns:
        np.ndarray: Bitmap with padding applied (1=original, 2=padding, 0=empty)
    """
    # Mode 0: No padding
    if mode == 0:
        return bitmap

    # Initialize padding array
    pad = np.zeros_like(bitmap)

    # Mode 1: Plus padding (all 4 orthogonal neighbors)
    if mode == 1:
        pad[:-1, :] += bitmap[1:, :]   # Top neighbor
        pad[1:, :] += bitmap[:-1, :]   # Bottom neighbor
        pad[:, :-1] += bitmap[:, 1:]   # Left neighbor
        pad[:, 1:] += bitmap[:, :-1]   # Right neighbor

    # Mode 2: Top-left padding (top, left, and top-left corner)
    elif mode == 2:
        pad[:-1, :] += bitmap[1:, :]   # Top neighbor
        pad[:, :-1] += bitmap[:, 1:]   # Left neighbor
        pad[:-1, :-1] += bitmap[1:, 1:]  # Top-left corner

    # Mode 3: Top-right padding (top, right, and top-right corner)
    elif mode == 3:
        pad[:-1, :] += bitmap[1:, :]   # Top neighbor
        pad[:, 1:] += bitmap[:, :-1]   # Right neighbor
        pad[:-1, 1:] += bitmap[1:, :-1]  # Top-right corner

    # Mode 4: Bottom-left padding (bottom, left, and bottom-left corner)
    elif mode == 4:
        pad[1:, :] += bitmap[:-1, :]   # Bottom neighbor
        pad[:, :-1] += bitmap[:, 1:]   # Left neighbor
        pad[1:, :-1] += bitmap[:-1, 1:]  # Bottom-left corner

    # Mode 5: Bottom-right padding (bottom, right, and bottom-right corner)
    elif mode == 5:
        pad[1:, :] += bitmap[:-1, :]   # Bottom neighbor
        pad[:, 1:] += bitmap[:, :-1]   # Right neighbor
        pad[1:, 1:] += bitmap[:-1, :-1]  # Bottom-right corner

    # Mode 6: Square padding (all 8 neighbors including diagonals)
    elif mode == 6:
        pad[:-1, :] += bitmap[1:, :]     # Top neighbor
        pad[1:, :] += bitmap[:-1, :]     # Bottom neighbor
        pad[:, :-1] += bitmap[:, 1:]     # Left neighbor
        pad[:, 1:] += bitmap[:, :-1]     # Right neighbor
        pad[:-1, :-1] += bitmap[1:, 1:]  # Top-left corner
        pad[:-1, 1:] += bitmap[1:, :-1]  # Top-right corner
        pad[1:, :-1] += bitmap[:-1, 1:]  # Bottom-left corner
        pad[1:, 1:] += bitmap[:-1, :-1]  # Bottom-right corner

    # Mark padding positions as 2, keep original tiles as 1
    pad = np.where(pad, 2, 0)
    pad = np.where(bitmap, 1, pad)
    return pad


# Test cases for mode 1 (plus padding)
res = _add_padding(np.array([[0, 1, 0],
                           [0, 0, 0],
                           [1, 0, 1]]), mode=1)
exp = np.array([[2, 1, 2],
              [2, 2, 2],
              [1, 2, 1]])
assert np.array_equal(res, exp), f"\n{res}\n{exp}"

res = _add_padding(np.array([[0, 0, 0],
                           [1, 0, 0],
                           [1, 0, 0]]), mode=1)
exp = np.array([[2, 0, 0],
              [1, 2, 0],
              [1, 2, 0]])
assert np.array_equal(res, exp), f"\n{res}\n{exp}"

# Test case for mode 0 (no padding)
res = _add_padding(np.array([[0, 1, 0],
                           [0, 0, 0],
                           [1, 0, 1]]), mode=0)
exp = np.array([[0, 1, 0],
              [0, 0, 0],
              [1, 0, 1]])
assert np.array_equal(res, exp), f"\n{res}\n{exp}"

# Test case for mode 2 (top-left padding)
res = _add_padding(np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]), mode=2)
exp = np.array([[2, 2, 0],
              [2, 1, 0],
              [0, 0, 0]])
assert np.array_equal(res, exp), f"\n{res}\n{exp}"

# Test case for mode 3 (top-right padding)
res = _add_padding(np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]), mode=3)
exp = np.array([[0, 2, 2],
              [0, 1, 2],
              [0, 0, 0]])
assert np.array_equal(res, exp), f"\n{res}\n{exp}"

# Test case for mode 4 (bottom-left padding)
res = _add_padding(np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]), mode=4)
exp = np.array([[0, 0, 0],
              [2, 1, 0],
              [2, 2, 0]])
assert np.array_equal(res, exp), f"\n{res}\n{exp}"

# Test case for mode 5 (bottom-right padding)
res = _add_padding(np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]), mode=5)
exp = np.array([[0, 0, 0],
              [0, 1, 2],
              [0, 2, 2]])
assert np.array_equal(res, exp), f"\n{res}\n{exp}"

# Test case for mode 6 (square padding - all 8 neighbors)
res = _add_padding(np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]), mode=6)
exp = np.array([[2, 2, 2],
              [2, 1, 2],
              [2, 2, 2]])
assert np.array_equal(res, exp), f"\n{res}\n{exp}"


def group_tiles(bitmap: np.ndarray, mode: int = 0) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """
    Original Python implementation of group_tiles (backup).

    Args:
        bitmap: 2D numpy array representing the grid of tiles
        mode: Padding mode (0-6)
              0 = No padding
              1 = Plus padding (all 4 orthogonal neighbors)
              2 = Top-left padding (top, left, and top-left corner)
              3 = Top-right padding (top, right, and top-right corner)
              4 = Bottom-left padding (bottom, left, and bottom-left corner)
              5 = Bottom-right padding (bottom, right, and bottom-right corner)
              6 = Square padding (all 8 neighbors including diagonals)

    Returns:
        list of tuples containing (mask, offset) for each polyomino
    """
    # Apply padding based on mode
    bitmap = _add_padding(bitmap, mode)

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
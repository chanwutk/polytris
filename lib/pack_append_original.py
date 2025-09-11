import numpy as np


def pack_append(
    poliominoes: list[tuple[int, np.ndarray, tuple[int, int]]],
    h: int,
    w: int,
    occupied_tiles: np.ndarray
) -> list[tuple[int, int, int, np.ndarray, tuple[int, int]]] | None:
    """
    Implementation of pack_append.
    
    Args:
        poliominoes: List of polyominoes to pack
        h: Height of the bitmap
        w: Width of the bitmap
        occupied_tiles: Existing bitmap to append to (modified in-place)
        
    Returns:
        list: positions where positions contains packing info
        or None if packing fails
    """
    appending_tiles = np.zeros((h, w), dtype=np.uint8)
    
    positions: list[tuple[int, int, int, np.ndarray, tuple[int, int]]] = []
    for groupid, mask, offset in poliominoes:
        for j in range(w - mask.shape[1] + 1):
            for i in range(h - mask.shape[0] + 1):
                if (
                    not np.any(occupied_tiles[i, j:j+mask.shape[1]] & mask[0]) and
                    (
                        mask.shape[0] == 1 or
                        not np.any(occupied_tiles[i+1:i+mask.shape[0], j:j+mask.shape[1]] & mask[1:])
                    )
                ):
                    occupied_tiles[i:i+mask.shape[0], j:j+mask.shape[1]] |= mask
                    appending_tiles[i:i+mask.shape[0], j:j+mask.shape[1]] |= mask
                    positions.append((i, j, groupid, mask, offset))
                    break
            else:
                continue
            break
        else:
            occupied_tiles ^= appending_tiles
            return None
    
    return positions
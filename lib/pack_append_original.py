import numpy as np


def pack_append(
    poliominoes: list[tuple[np.ndarray, tuple[int, int]]],
    h: int,
    w: int,
    occupied_tiles: np.ndarray
) -> list[tuple[int, int, np.ndarray, tuple[int, int]]] | None:
    """
    Pack all polyominoes into the occupied tiles.
    If not possible, return None.
    Note:
        - occupied_tiles will be reverted to the original state if packing fails.
    
    Args:
        poliominoes: List of polyominoes to pack
        h: Height of the occupied_tiles
        w: Width of the occupied_tiles
        occupied_tiles: Existing bitmap to append polyominoes to (modified in-place)
        
    Returns:
        list: positions where each position contains (i, j, mask, offset) or None if packing fails
            where:
                - i: y-coordinate of the position
                - j: x-coordinate of the position
                - mask: masking of the polyomino as a numpy array,
                - offset: offset of the mask of its original position
    """
    appending_tiles = np.zeros((h, w), dtype=np.uint8)
    
    positions: list[tuple[int, int, np.ndarray, tuple[int, int]]] = []
    for mask, offset in poliominoes:
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
                    positions.append((i, j, mask, offset))
                    break
            else:
                continue
            break
        else:
            occupied_tiles ^= appending_tiles
            return None
    
    return positions
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

cimport numpy as cnp
import cython
from libc.stdlib cimport malloc, free, qsort

import numpy as np

from polyis.binpack.utilities cimport IntStack, Polyomino, PolyominoStack, \
                       IntStack_init, IntStack_push, IntStack_cleanup


# PolyominoPosition represents the position and orientation of a polyomino in a collage
# Attributes:
#   oy: Original y-coordinate offset of the polyomino from its video frame
#   ox: Original x-coordinate offset of the polyomino from its video frame
#   py: Packed y-coordinate position in the collage
#   px: Packed x-coordinate position in the collage
#   rotation: Rotation applied to the polyomino (0-3, representing 0°, 90°, 180°, 270°)
#   frame: Frame index this polyomino belongs to
#   shape: The actual polyomino shape as a numpy array
cdef class PolyominoPosition:
    cdef public int oy
    cdef public int ox
    cdef public int py
    cdef public int px
    cdef public int rotation
    cdef public int frame
    cdef public object shape  # numpy array
    
    def __init__(self, int oy, int ox, int py, int px, int rotation, int frame, object shape):
        self.oy = oy
        self.ox = ox
        self.py = py
        self.px = px
        self.rotation = rotation
        self.frame = frame
        self.shape = shape


# Placement represents a successful placement of a polyomino in a collage
# Attributes:
#   y: Y-coordinate where the polyomino was placed
#   x: X-coordinate where the polyomino was placed
#   rotation: Rotation applied to the polyomino (0-3)
cdef class Placement:
    cdef public int y
    cdef public int x
    cdef public int rotation
    
    def __init__(self, int y, int x, int rotation):
        self.y = y
        self.x = x
        self.rotation = rotation


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int large_first(const void *a, const void *b) noexcept nogil:
    """
    Comparison function for qsort to sort polyominoes by mask length (descending order).
    Returns negative if a should come before b, positive if b should come before a.
    """
    # Compare by mask length (top field of IntStack) in descending order
    # Larger masks first (negative return means a comes before b)
    return (<Polyomino*>b).mask.top - (<Polyomino*>a).mask.top


# @cython.boundscheck(False)  # type: ignore
# @cython.wraparound(False)  # type: ignore
# @cython.nonecheck(False)  # type: ignore
# cdef cnp.uint8_t[:, :] rotate_polyomino(cnp.uint8_t[:, :] polyomino, int rotation):
#     """
#     Rotate a polyomino by the specified number of 90-degree counter-clockwise rotations.
    
#     Args:
#         polyomino: 2D numpy array representing the polyomino shape
#         rotation: Number of 90-degree rotations (0-3)
        
#     Returns:
#         Rotated polyomino as a new numpy array
#     """
#     cdef int h = polyomino.shape[0]
#     cdef int w = polyomino.shape[1]
#     cdef int new_h, new_w
#     cdef int i, j
    
#     # Determine new dimensions based on rotation
#     if rotation % 2 == 0:
#         new_h, new_w = h, w
#     else:
#         new_h, new_w = w, h
    
#     # Create new array for rotated polyomino
#     cdef cnp.uint8_t[:, :] rotated = np.zeros((new_h, new_w), dtype=np.uint8)
    
#     # Apply rotation
#     for i in range(h):
#         for j in range(w):
#             if polyomino[i, j]:  # type: ignore
#                 if rotation == 0:
#                     rotated[i, j] = 1  # type: ignore
#                 elif rotation == 1:  # 90 degrees counter-clockwise
#                     rotated[j, h - 1 - i] = 1  # type: ignore
#                 elif rotation == 2:  # 180 degrees
#                     rotated[h - 1 - i, w - 1 - j] = 1  # type: ignore
#                 elif rotation == 3:  # 270 degrees counter-clockwise
#                     rotated[w - 1 - j, i] = 1  # type: ignore
    
#     return rotated


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef Placement try_pack(cnp.uint8_t[:, :] polyomino, cnp.uint8_t[:, :] occupied_tiles):
    """
    Attempts to place a polyomino in a collage by trying all rotations and positions.
    
    This function tries to find a valid placement for the given polyomino in the collage
    by testing all 4 possible rotations (0°, 90°, 180°, 270°) and all possible positions
    where the polyomino would fit within the collage boundaries.
    
    Args:
        polyomino: 2D numpy array representing the polyomino shape (1s for occupied cells, 0s for empty)
        occupied_tiles: 2D numpy array representing the current collage state (0s for empty cells, 1s for occupied)
        
    Returns:
        Placement object with successful position and rotation if found, None otherwise.
        The occupied_tiles is modified in-place when a successful placement is found.
    """
    cdef cnp.uint8_t[:, :] rotated
    cdef int ph, pw, ch, cw
    cdef int y, x, rotation
    cdef int i, j
    cdef bint overlap

    rotation = 0
    
    # # Try all 4 possible rotations (0°, 90°, 180°, 270°)
    # for rotation in range(4):
    # Rotate the polyomino by 90 degrees * rotation counter-clockwise
    # rotated = rotate_polyomino(polyomino, rotation)
    rotated = polyomino
    # Get dimensions of the rotated polyomino
    ph = rotated.shape[0]
    pw = rotated.shape[1]
    # Get dimensions of the collage
    ch = occupied_tiles.shape[0]
    cw = occupied_tiles.shape[1]

    # Try all possible positions where the polyomino would fit
    for y in range(ch - ph + 1):
        for x in range(cw - pw + 1):
            # Check if there's no overlap between existing collage and polyomino
            overlap = <bint>False
            for i in range(ph):
                for j in range(pw):
                    if rotated[i, j] and occupied_tiles[y + i, x + j]:  # type: ignore
                        overlap = <bint>True
                        break
                if overlap:
                    break
            
            if not overlap:
                # Place the polyomino by adding it to the collage
                for i in range(ph):
                    for j in range(pw):
                        if rotated[i, j]:  # type: ignore
                            occupied_tiles[y + i, x + j] = 1  # type: ignore
                
                # Return the successful placement coordinates and rotation
                return Placement(y, x, rotation)
    
    # No valid placement found
    return NULL


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def pack_all(list polyominoes_stacks, int h, int w) -> list:
    """
    Packs all polyominoes from multiple stacks into collages using a bin packing algorithm.

    This function takes multiple stacks of polyominoes (memory addresses) and attempts to
    pack them into rectangular collages of the specified dimensions. It uses a greedy
    first-fit decreasing algorithm, trying to place the largest polyominoes first.
    
    Args:
        polyominoes_stacks: List of integers, each representing a stack of polyominoes
                            a stack of polyominoes is a memory address pointing to a
                            stack of polyominoes from Cython.
                            Each stack of polyominoes corresponds to a video frame.
        h: Height of each collage in pixels
        w: Width of each collage in pixels
        
    Returns:
        List of lists, where each inner list represents a collage, which contains
        PolyominoPosition objects representing all polyominoes packed into a single collage.
    """
    cdef unsigned long long polyominoes_stack_ptr
    cdef PolyominoStack *polyominoes_stack
    cdef int stack_idx, poly_idx, i, j, k
    cdef Polyomino *all_polyominoes
    cdef int *all_frames
    cdef Polyomino polyomino
    cdef int count_polyominos = 0
    cdef int num_stacks
    cdef int num_polyominos
    cdef int total_polyominos = 0
    cdef IntStack mask
    cdef unsigned short max_i, max_j, tile_i, tile_j
    cdef int mask_h, mask_w
    cdef cnp.uint8_t[:, :] shape
    cdef int oy, ox
    cdef bint placed
    cdef Placement res
    cdef int py, px, rotation
    cdef PolyominoPosition pos
    cdef cnp.uint8_t[:, :] collage
    
    # Initialize lists to store frame indices and polyomino data
    cdef list all_frames_list = []
    cdef list all_polyominoes_list = []
    
    # Combine stacks of polyominoes into a single list of polyominoes with frame indices
    num_stacks = len(polyominoes_stacks)
    for i in range(num_stacks):
        polyominoes_stack_ptr = polyominoes_stacks[i]
        polyominoes_stack = <PolyominoStack*>polyominoes_stack_ptr
        num_polyominos = polyominoes_stack.top
        total_polyominos += num_polyominos
        
        # Convert each polyomino to numpy array format
        for poly_idx in range(num_polyominos):
            polyomino = polyominoes_stack.mo_data[poly_idx]  # type: ignore
            
            # Convert IntStack mask to numpy array
            mask = polyomino.mask
            max_i = 0
            max_j = 0
            
            # Find dimensions
            for k in range(mask.top // 2):
                tile_i = mask.data[k << 1]  # type: ignore
                tile_j = mask.data[(k << 1) + 1]  # type: ignore
                if tile_i > max_i:
                    max_i = tile_i
                if tile_j > max_j:
                    max_j = tile_j
            
            mask_h = max_i + 1
            mask_w = max_j + 1
            shape = np.zeros((mask_h, mask_w), dtype=np.uint8)
            
            # Fill the shape array
            for k in range(mask.top // 2):
                tile_i = mask.data[k << 1]  # type: ignore
                tile_j = mask.data[(k << 1) + 1]  # type: ignore
                shape[tile_i, tile_j] = 1  # type: ignore
            
            # Record which frame each polyomino belongs to
            all_frames_list.append(i)
            all_polyominoes_list.append((shape, (polyomino.offset_i, polyomino.offset_j)))
    
    # If no polyominoes, return empty result
    if total_polyominos == 0:
        return []
    
    # Sort polyominoes by size (largest first) for better packing efficiency
    all_polyominoes_frames = list(zip(all_polyominoes_list, all_frames_list))
    all_polyominoes_frames.sort(key=lambda x: -np.sum(x[0][0]))
    
    # Initialize storage for collages and their corresponding polyomino positions
    cdef list collages_pool = []
    cdef list positions = []
    
    # Process each polyomino in size order (largest first)
    for polyomino_data, frame in all_polyominoes_frames:
        # Extract the shape and original offset coordinates
        shape, (oy, ox) = polyomino_data
        
        # Try to place the polyomino in an existing collage
        placed = False
        for i, collage in enumerate(collages_pool):
            # Attempt to pack the polyomino in this collage
            res = try_pack(shape, collage)
            if res is not None:
                # Successfully placed - extract position and rotation
                py = res.y
                px = res.x
                rotation = res.rotation
                
                # Record the polyomino position in this collage
                pos = PolyominoPosition(oy, ox, py, px, rotation, frame, shape)
                positions[i].append(pos)
                placed = True
                break
        
        if not placed:
            # No existing collage could fit this polyomino - create a new collage
            # Create a new empty collage with specified dimensions
            collage = np.zeros((h, w), dtype=np.uint8)
            # Add the new collage to the pool
            collages_pool.append(collage)
            # Attempt to place the polyomino in the new collage
            res = try_pack(shape, collage)
            # This should always succeed since the collage is empty
            if res is not None:
                # Extract position and rotation from successful placement
                py = res.y
                px = res.x
                rotation = res.rotation
                
                # Create a new positions list for this collage with the first polyomino
                pos = PolyominoPosition(oy, ox, py, px, rotation, frame, shape)
                positions.append([pos])
    
    # Return all collages with their packed polyomino positions
    return positions


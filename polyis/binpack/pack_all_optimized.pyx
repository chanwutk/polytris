# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

cimport numpy as cnp
import numpy as np
import cython
from libc.stdlib cimport malloc, free, calloc, qsort

from polyis.binpack.utilities cimport (
    IntStack, Polyomino, PolyominoStack,
    IntStack_init, IntStack_push, IntStack_cleanup,
    PolyominoStack_init, PolyominoStack_push, PolyominoStack_cleanup,
    Polyomino_cleanup
)


# Data Structures

cdef struct PolyominoPosition:
    """Represents the position and orientation of a polyomino in a collage."""
    int oy  # Original y-coordinate offset of the polyomino from its video frame
    int ox  # Original x-coordinate offset of the polyomino from its video frame
    int py  # Packed y-coordinate position in the collage
    int px  # Packed x-coordinate position in the collage
    int rotation  # Rotation applied to the polyomino (0-3, representing 0°, 90°, 180°, 270°)
    int frame  # Frame index this polyomino belongs to
    object shape  # The actual polyomino shape stored as numpy array for Python compatibility


cdef struct Placement:
    """Represents a successful placement of a polyomino in a collage."""
    int y  # Y-coordinate where the polyomino was placed
    int x  # X-coordinate where the polyomino was placed
    int rotation  # Rotation applied to the polyomino (0-3)


cdef struct CollageMetadata:
    """Holds a collage and cached metadata about its unoccupied regions."""
    object occupied_tiles  # 2D numpy array representing the collage (0=empty, 1=occupied)
    object unoccupied_spaces  # List of numpy array masks, each representing one unoccupied region
    object space_sizes  # List of integers, parallel to unoccupied_spaces, storing the size of each space


# Python wrapper classes for return values
class PolyominoPositionWrapper:
    """Python wrapper for PolyominoPosition struct."""
    def __init__(self, int oy, int ox, int py, int px, int rotation, int frame, object shape):
        self.oy = oy
        self.ox = ox
        self.py = py
        self.px = px
        self.rotation = rotation
        self.frame = frame
        self.shape = shape


cdef object _make_polyomino_position(PolyominoPosition pos):
    """Convert PolyominoPosition struct to Python object."""
    return PolyominoPositionWrapper(pos.oy, pos.ox, pos.py, pos.px, pos.rotation, pos.frame, pos.shape)


# Helper Functions for Connected Components Labeling

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef IntStack _find_connected_empty_tiles(
    unsigned int* visited,
    unsigned int region_id,
    int h,
    int w,
    int start_i,
    int start_j,
    cnp.uint8_t[:, :] occupied_tiles
) noexcept nogil:
    """
    Groups connected empty tiles into a region using flood fill.
    
    This function modifies the visited array in-place to mark visited tiles.
    The algorithm uses flood fill: it starts with a unique value at (start_i, start_j)
    and spreads that value to all connected empty tiles, collecting their positions.
    
    Args:
        visited: 2D memoryview stored as 1D array of the visited bitmap (modified in-place)
        region_id: Unique ID for this region
        h: Height of the grid
        w: Width of the grid
        start_i: Starting row index
        start_j: Starting column index
        occupied_tiles: 2D numpy array memoryview (uint8) representing the grid of tiles,
                       where 0 indicates empty and 1 indicates occupied
    
    Returns:
        IntStack: IntStack containing coordinate pairs for connected empty tiles
    """
    cdef unsigned short i, j, _i, _j
    cdef int di
    cdef IntStack filled, stack
    cdef char[4] DIRECTIONS_I = [-1, 0, 1, 0]  # type: ignore
    cdef char[4] DIRECTIONS_J = [0, -1, 0, 1]  # type: ignore
    
    if IntStack_init(&filled, 16):
        IntStack_cleanup(&filled)
        return filled
    
    if IntStack_init(&stack, 16):
        IntStack_cleanup(&stack)
        IntStack_cleanup(&filled)
        return filled
    
    # Push initial coordinates
    IntStack_push(&stack, <unsigned short>start_i)
    IntStack_push(&stack, <unsigned short>start_j)
    
    while stack.top > 0:
        j = stack.data[stack.top - 1]  # type: ignore
        i = stack.data[stack.top - 2]  # type: ignore
        stack.top -= 2
        
        # Mark current position as visited and add to result
        visited[i * w + j] = region_id  # type: ignore
        IntStack_push(&filled, i)
        IntStack_push(&filled, j)
        
        # Check all 4 directions for unvisited connected empty tiles
        for di in range(4):
            _i = i + DIRECTIONS_I[di]  # type: ignore
            _j = j + DIRECTIONS_J[di]  # type: ignore
            
            if 0 <= _i < h and 0 <= _j < w:
                # Check if tile is empty and not visited
                if occupied_tiles[_i, _j] == 0 and visited[_i * w + _j] == 0:  # type: ignore
                    IntStack_push(&stack, <unsigned short>_i)
                    IntStack_push(&stack, <unsigned short>_j)
    
    # Free the stack's data before returning
    IntStack_cleanup(&stack)
    return filled


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef object _intstack_to_numpy_array(IntStack* mask, int h, int w):
    """Convert IntStack mask to numpy array of same shape as occupied_tiles."""
    cdef object mask_array_obj = np.zeros((h, w), dtype=np.uint8)
    cdef cnp.uint8_t[:, :] mask_array = mask_array_obj  # type: ignore
    cdef int k
    cdef unsigned short tile_i, tile_j
    for k in range(mask.top // 2):
        tile_i = mask.data[k << 1]  # type: ignore
        tile_j = mask.data[(k << 1) + 1]  # type: ignore
        if tile_i < h and tile_j < w:
            mask_array[tile_i, tile_j] = 1  # type: ignore
    return mask_array_obj


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def extract_unoccupied_spaces(cnp.uint8_t[:, :] occupied_tiles):
    """Extract all unoccupied connected regions as numpy array masks and their sizes.
    
    Args:
        occupied_tiles: 2D numpy array representing the collage state
    
    Returns:
        Tuple of (list of numpy array masks, list of sizes)
        Each mask represents one unoccupied region (same shape as occupied_tiles, 1=unoccupied in this region)
        Each size is the number of tiles in the corresponding region
    """
    cdef int h = occupied_tiles.shape[0]
    cdef int w = occupied_tiles.shape[1]
    cdef unsigned int* visited = <unsigned int*>calloc(<size_t>(h * w), <size_t>sizeof(unsigned int))
    cdef unsigned int region_id = 1
    cdef int i, j
    cdef IntStack region_mask
    cdef int region_size
    cdef list unoccupied_spaces = []
    cdef list space_sizes = []
    
    # Process each cell to find unoccupied connected regions
    for i in range(h):
        for j in range(w):
            # Skip if occupied or already visited
            if occupied_tiles[i, j] != 0 or visited[i * w + j] != 0:  # type: ignore
                continue
            
            # Find connected empty tiles for this region
            region_mask = _find_connected_empty_tiles(visited, region_id, h, w, i, j, occupied_tiles)
            if region_mask.top > 0:
                # Calculate size (number of coordinate pairs = top / 2)
                region_size = region_mask.top // 2
                # Convert IntStack to numpy array for storage
                mask_array = _intstack_to_numpy_array(&region_mask, h, w)
                unoccupied_spaces.append(mask_array)
                space_sizes.append(region_size)
                IntStack_cleanup(&region_mask)
                region_id += 1
    
    free(<void*>visited)
    return unoccupied_spaces, space_sizes


# Sparse Polyomino Operations

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef bint check_collision(
    Polyomino* polyomino,
    int placement_y,
    int placement_x,
    cnp.uint8_t[:, :] occupied_tiles
) noexcept nogil:
    """
    Check if placing a polyomino at the given position would cause a collision.
    
    Args:
        polyomino: Pointer to Polyomino structure with IntStack mask
        placement_y: Y-coordinate where to place the polyomino
        placement_x: X-coordinate where to place the polyomino
        occupied_tiles: 2D array representing the collage state
    
    Returns:
        True if collision detected, False otherwise
    """
    cdef int k
    cdef unsigned short tile_i, tile_j
    cdef int abs_y, abs_x
    cdef IntStack mask = polyomino.mask
    cdef int h = occupied_tiles.shape[0]
    cdef int w = occupied_tiles.shape[1]
    
    # Check each tile in the polyomino mask
    for k in range(mask.top // 2):
        tile_i = mask.data[k << 1]  # type: ignore
        tile_j = mask.data[(k << 1) + 1]  # type: ignore
        abs_y = placement_y + tile_i
        abs_x = placement_x + tile_j
        
        # Check bounds
        if abs_y < 0 or abs_y >= h or abs_x < 0 or abs_x >= w:
            return <bint>True
        
        # Check collision
        if occupied_tiles[abs_y, abs_x] != 0:  # type: ignore
            return <bint>True
    
    return <bint>False


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void place_polyomino(
    Polyomino* polyomino,
    int placement_y,
    int placement_x,
    cnp.uint8_t[:, :] occupied_tiles
) noexcept nogil:
    """
    Place a polyomino at the given position by setting occupied tiles.
    
    Args:
        polyomino: Pointer to Polyomino structure with IntStack mask
        placement_y: Y-coordinate where to place the polyomino
        placement_x: X-coordinate where to place the polyomino
        occupied_tiles: 2D array representing the collage state (modified in-place)
    """
    cdef int k
    cdef unsigned short tile_i, tile_j
    cdef int abs_y, abs_x
    cdef IntStack mask = polyomino.mask
    
    # Set each tile in the polyomino mask as occupied
    for k in range(mask.top // 2):
        tile_i = mask.data[k << 1]  # type: ignore
        tile_j = mask.data[(k << 1) + 1]  # type: ignore
        abs_y = placement_y + tile_i
        abs_x = placement_x + tile_j
        occupied_tiles[abs_y, abs_x] = 1  # type: ignore


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef Placement* try_pack(
    Polyomino* polyomino,
    cnp.uint8_t[:, :] occupied_tiles
):
    """
    Attempts to place a polyomino in a collage by trying all positions.
    
    This function tries to find a valid placement for the given polyomino in the collage
    by testing all possible positions where the polyomino would fit within the collage boundaries.
    
    Args:
        polyomino: Pointer to Polyomino structure with IntStack mask
        occupied_tiles: 2D numpy array representing the current collage state (0s for empty cells, 1s for occupied)
    
    Returns:
        Placement object with successful position and rotation if found, NULL otherwise.
        The occupied_tiles is modified in-place when a successful placement is found.
    """
    cdef int h = occupied_tiles.shape[0]
    cdef int w = occupied_tiles.shape[1]
    cdef IntStack mask = polyomino.mask
    cdef int max_i = 0, max_j = 0
    cdef int k
    cdef unsigned short tile_i, tile_j
    cdef int ph, pw
    cdef int y, x
    
    # Calculate bounding box of polyomino
    for k in range(mask.top // 2):
        tile_i = mask.data[k << 1]  # type: ignore
        tile_j = mask.data[(k << 1) + 1]  # type: ignore
        if tile_i > max_i:
            max_i = tile_i
        if tile_j > max_j:
            max_j = tile_j
    
    ph = max_i + 1
    pw = max_j + 1
    
    # Try all possible positions where the polyomino would fit
    for y in range(h - ph + 1):
        for x in range(w - pw + 1):
            # Check for collision
            if not check_collision(polyomino, y, x, occupied_tiles):
                # Place the polyomino
                place_polyomino(polyomino, y, x, occupied_tiles)
                # Return successful placement
                cdef Placement* result = <Placement*>malloc(sizeof(Placement))
                result.y = y
                result.x = x
                result.rotation = 0
                return result
    
    # No valid placement found
    return <Placement*>NULL


# Collage Metadata Operations

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int count_regions_at_least(CollageMetadata collage_meta, int min_size):
    """Count how many unoccupied regions are at least the given size.
    
    Args:
        collage_meta: The CollageMetadata to query
        min_size: Minimum size threshold
    
    Returns:
        Count of regions with size >= min_size
    """
    cdef int count = 0
    cdef int size
    cdef list space_sizes_list = <list>collage_meta.space_sizes
    for size in space_sizes_list:
        if size >= min_size:
            count += 1
    return count


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef CollageMetadata update_collage_after_placement(
    CollageMetadata collage_meta,
    Placement* placement,
    Polyomino* polyomino
):
    """
    Update cached region information after a polyomino has been placed.
    
    Finds which unoccupied space contains the placement location and updates only that space.
    The placed polyomino breaks one unoccupied region into potentially smaller ones.
    
    Args:
        collage_meta: The CollageMetadata to update
        placement: Placement object containing the position where the polyomino was placed
        polyomino: Pointer to Polyomino structure that was placed
    
    Returns:
        New CollageMetadata with updated unoccupied spaces
    """
    cdef int py = placement.y
    cdef int px = placement.x
    cdef IntStack mask = polyomino.mask
    cdef object occupied_tiles_obj = collage_meta.occupied_tiles
    cdef cnp.uint8_t[:, :] occupied_tiles = occupied_tiles_obj
    cdef int h = occupied_tiles.shape[0]
    cdef int w = occupied_tiles.shape[1]
    
    # Find which unoccupied space contains the placement location
    # Find one tile that is part of the polyomino shape (guaranteed to be in exactly one unoccupied space)
    cdef unsigned short tile_rel_y, tile_rel_x
    if mask.top < 2:
        raise ValueError("Polyomino has no tiles")
    
    # Get the first tile's coordinates relative to the polyomino shape
    tile_rel_y = mask.data[0]  # type: ignore
    tile_rel_x = mask.data[1]  # type: ignore
    # Convert to absolute coordinates in the collage
    cdef int tile_abs_y = py + tile_rel_y
    cdef int tile_abs_x = px + tile_rel_x
    
    # Find which unoccupied space contains this tile
    cdef int affected_space_idx = -1
    cdef int idx
    cdef cnp.uint8_t[:, :] space_mask
    cdef object space_mask_obj
    cdef list unoccupied_spaces_list = <list>collage_meta.unoccupied_spaces
    
    for idx in range(len(unoccupied_spaces_list)):
        space_mask_obj = unoccupied_spaces_list[idx]
        space_mask = <cnp.uint8_t[:, :]>space_mask_obj  # type: ignore
        # Check if this tile is in this unoccupied space
        if tile_abs_y < space_mask.shape[0] and tile_abs_x < space_mask.shape[1] and space_mask[tile_abs_y, tile_abs_x] > 0:  # type: ignore
            affected_space_idx = idx
            break
    
    # This should always succeed since the polyomino was successfully placed
    if affected_space_idx == -1:
        raise ValueError("Failed to find the affected unoccupied space")
    
    # Get the affected unoccupied space mask
    cdef object affected_space_obj = unoccupied_spaces_list[affected_space_idx]
    cdef cnp.uint8_t[:, :] affected_space = <cnp.uint8_t[:, :]>affected_space_obj  # type: ignore
    
    # Create a mask for the placed polyomino in the full collage coordinates
    cdef object placed_polyomino_mask_obj = np.zeros((h, w), dtype=np.uint8)
    cdef cnp.uint8_t[:, :] placed_polyomino_mask = placed_polyomino_mask_obj  # type: ignore
    cdef unsigned short tile_rel_y, tile_rel_x
    cdef int abs_y, abs_x
    cdef int k
    
    # Fill the placed polyomino mask
    for k in range(mask.top // 2):
        tile_rel_y = mask.data[k << 1]  # type: ignore
        tile_rel_x = mask.data[(k << 1) + 1]  # type: ignore
        abs_y = py + tile_rel_y
        abs_x = px + tile_rel_x
        if abs_y < h and abs_x < w:
            placed_polyomino_mask[abs_y, abs_x] = 1  # type: ignore
    
    # Subtract the placed polyomino from the affected space
    # Use int16 for subtraction to avoid underflow, then convert back to uint8
    cdef object remaining_space_obj = np.zeros((h, w), dtype=np.int16)
    cdef cnp.int16_t[:, :] remaining_space = remaining_space_obj
    cdef int i, j
    cdef cnp.int16_t val
    
    # Copy affected_space and subtract placed polyomino
    for i in range(h):
        for j in range(w):
            val = <cnp.int16_t>affected_space[i, j] - <cnp.int16_t>placed_polyomino_mask[i, j]  # type: ignore
            remaining_space[i, j] = val  # type: ignore
    
    # Convert back to uint8 (values should be 0 or 1)
    cdef object remaining_space_uint8 = np.clip(remaining_space_obj, 0, 1).astype(np.uint8)
    cdef cnp.uint8_t[:, :] remaining_space_final = remaining_space_uint8
    
    # Extract new connected components from the remaining space
    cdef list new_spaces
    cdef list new_space_sizes
    new_spaces, new_space_sizes = extract_unoccupied_spaces(remaining_space_final)
    
    # Build updated lists: keep unaffected spaces, replace affected space with new spaces
    cdef list space_sizes_list = <list>collage_meta.space_sizes
    cdef list updated_unoccupied_spaces = (
        unoccupied_spaces_list[:affected_space_idx] + 
        new_spaces + 
        unoccupied_spaces_list[affected_space_idx + 1:]
    )
    cdef list updated_space_sizes = (
        space_sizes_list[:affected_space_idx] + 
        new_space_sizes + 
        space_sizes_list[affected_space_idx + 1:]
    )
    
    cdef CollageMetadata result
    result.occupied_tiles = collage_meta.occupied_tiles
    result.unoccupied_spaces = updated_unoccupied_spaces
    result.space_sizes = updated_space_sizes
    return result


# Helper function to deep copy IntStack

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void IntStack_copy(IntStack* dest, IntStack* src) noexcept nogil:
    """Deep copy an IntStack."""
    cdef int i
    IntStack_init(dest, src.capacity)
    for i in range(src.top):
        IntStack_push(dest, src.data[i])  # type: ignore


# Main Packing Function

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int compare_polyomino_by_size(const void *a, const void *b) noexcept nogil:
    """
    Comparison function for qsort to sort polyominoes by mask size (descending order).
    Returns negative if a should come before b, positive if b should come before a.
    """
    # Compare by mask size (top field of IntStack) in descending order
    # Larger masks first (negative return means a comes before b)
    return (<PolyominoWithFrame*>b).polyomino.mask.top - (<PolyominoWithFrame*>a).polyomino.mask.top


cdef struct PolyominoWithFrame:
    Polyomino polyomino
    int frame
    int offset_i
    int offset_j


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def pack_all(list polyominoes_stacks, int h, int w):
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
    cdef int num_stacks = len(polyominoes_stacks)
    cdef int stack_idx, poly_idx
    cdef Polyomino polyomino
    cdef int total_polyominos = 0
    cdef int num_polyominos
    cdef PolyominoWithFrame* all_polyominoes
    cdef int count = 0
    cdef int i
    
    # First pass: count total polyominoes
    for stack_idx in range(num_stacks):
        polyominoes_stack_ptr = polyominoes_stacks[stack_idx]
        polyominoes_stack = <PolyominoStack*>polyominoes_stack_ptr
        total_polyominos += polyominoes_stack.top
    
    if total_polyominos == 0:
        return []
    
    # Allocate array for all polyominoes with frame information
    all_polyominoes = <PolyominoWithFrame*>malloc(<size_t>total_polyominos * sizeof(PolyominoWithFrame))
    
    # Second pass: collect all polyominoes with frame indices
    for stack_idx in range(num_stacks):
        polyominoes_stack_ptr = polyominoes_stacks[stack_idx]
        polyominoes_stack = <PolyominoStack*>polyominoes_stack_ptr
        num_polyominos = polyominoes_stack.top
        
        for poly_idx in range(num_polyominos):
            polyomino = polyominoes_stack.mo_data[poly_idx]  # type: ignore
            # Deep copy the IntStack mask to avoid issues with qsort
            IntStack_copy(&all_polyominoes[count].polyomino.mask, &polyomino.mask)
            all_polyominoes[count].polyomino.offset_i = polyomino.offset_i
            all_polyominoes[count].polyomino.offset_j = polyomino.offset_j
            all_polyominoes[count].frame = stack_idx
            all_polyominoes[count].offset_i = polyomino.offset_i
            all_polyominoes[count].offset_j = polyomino.offset_j
            count += 1
    
    # Sort polyominoes by size (largest first) for better packing efficiency
    qsort(<void*>all_polyominoes,
          <size_t>total_polyominos,
          <size_t>sizeof(PolyominoWithFrame),
          &compare_polyomino_by_size)
    
    # Initialize storage for collages (with cached metadata) and their corresponding polyomino positions
    cdef list collages_pool = []
    cdef list positions = []
    cdef PolyominoWithFrame* current_poly
    cdef Polyomino* poly_ptr
    cdef int polyomino_size
    cdef int empty_space
    cdef int num_fitting_regions
    cdef list collage_candidates = []
    cdef tuple candidate_tuple
    cdef CollageMetadata collage_meta
    cdef CollageMetadata* collage_meta_ptr
    cdef Placement* res
    cdef int py, px, rotation
    cdef object shape_np_array  # Convert IntStack to numpy array for PolyominoPosition
    cdef int max_i, max_j, k
    cdef unsigned short tile_i, tile_j
    cdef int mask_h, mask_w
    cdef cnp.uint8_t[:, :] shape_view
    cdef int j
    cdef object unoccupied_spaces
    cdef list space_sizes
    cdef PolyominoPosition pos_struct
    
    # Process each polyomino in size order (largest first)
    for i in range(total_polyominos):
        current_poly = &all_polyominoes[i]
        poly_ptr = &current_poly.polyomino
        
        # Calculate the size of the polyomino (number of tiles = mask.top / 2)
        polyomino_size = poly_ptr.mask.top // 2
        
        # Try to place the polyomino in an existing collage
        # Evaluate each collage by counting unoccupied regions larger than the polyomino
        collage_candidates = []
        for j in range(len(collages_pool)):
            collage_meta_ptr = <CollageMetadata*>collages_pool[j]
            collage_meta = collage_meta_ptr[0]
            # First, check if there's enough total empty space
            empty_space = np.sum(collage_meta.occupied_tiles == 0)
            if empty_space >= polyomino_size:
                # Use cached region information to count fitting regions efficiently
                num_fitting_regions = count_regions_at_least(collage_meta, polyomino_size)
                if num_fitting_regions > 0:
                    collage_candidates.append((j, num_fitting_regions))
        
        # Sort by number of fitting unoccupied regions (descending order)
        collage_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Try to pack in collages with the most fitting unoccupied regions first
        cdef bint placed = <bint>False
        for candidate_tuple in collage_candidates:
            j = candidate_tuple[0]
            collage_meta_ptr = <CollageMetadata*>collages_pool[j]
            collage_meta = collage_meta_ptr[0]
            # Attempt to pack the polyomino in this collage
            cdef object occupied_tiles_obj = collage_meta.occupied_tiles
            cdef cnp.uint8_t[:, :] occupied_tiles_view = occupied_tiles_obj  # type: ignore
            res = try_pack(poly_ptr, occupied_tiles_view)
            if res != NULL:
                # Successfully placed - extract position and rotation
                py = res.y
                px = res.x
                rotation = res.rotation
                
                # Convert IntStack mask to numpy array for PolyominoPosition.shape
                # Calculate bounding box
                max_i = 0
                max_j = 0
                for k in range(poly_ptr.mask.top // 2):
                    tile_i = poly_ptr.mask.data[k << 1]  # type: ignore
                    tile_j = poly_ptr.mask.data[(k << 1) + 1]  # type: ignore
                    if tile_i > max_i:
                        max_i = tile_i
                    if tile_j > max_j:
                        max_j = tile_j
                
                mask_h = max_i + 1
                mask_w = max_j + 1
                shape_np_array = np.zeros((mask_h, mask_w), dtype=np.uint8)
                shape_view = shape_np_array
                for k in range(poly_ptr.mask.top // 2):
                    tile_i = poly_ptr.mask.data[k << 1]  # type: ignore
                    tile_j = poly_ptr.mask.data[(k << 1) + 1]  # type: ignore
                    shape_view[tile_i, tile_j] = 1  # type: ignore
                
                # Record the polyomino position in this collage
                pos_struct.oy = current_poly.offset_i
                pos_struct.ox = current_poly.offset_j
                pos_struct.py = py
                pos_struct.px = px
                pos_struct.rotation = rotation
                pos_struct.frame = current_poly.frame
                pos_struct.shape = shape_np_array
                positions[j].append(_make_polyomino_position(pos_struct))
                # Update cached region information for this collage only
                collage_meta = update_collage_after_placement(collage_meta, res, poly_ptr)
                collage_meta_ptr = <CollageMetadata*>malloc(sizeof(CollageMetadata))
                if collage_meta_ptr == NULL:
                    raise MemoryError("Failed to allocate CollageMetadata")
                collage_meta_ptr[0] = collage_meta  # type: ignore
                collages_pool[j] = <object>collage_meta_ptr
                placed = <bint>True
                free(res)
                break
        
        if not placed:
            # No existing collage could fit this polyomino - create a new collage
            # Create a new empty collage with specified dimensions
            cdef cnp.uint8_t[:, :] collage_array = np.zeros((h, w), dtype=np.uint8)
            # Attempt to place the polyomino in the new collage
            res = try_pack(poly_ptr, collage_array)
            # This should always succeed since the collage is empty
            if res != NULL:
                # Extract position and rotation from successful placement
                py = res.y
                px = res.x
                rotation = res.rotation
                
                # Convert IntStack mask to numpy array for PolyominoPosition.shape
                max_i = 0
                max_j = 0
                for k in range(poly_ptr.mask.top // 2):
                    tile_i = poly_ptr.mask.data[k << 1]  # type: ignore
                    tile_j = poly_ptr.mask.data[(k << 1) + 1]  # type: ignore
                    if tile_i > max_i:
                        max_i = tile_i
                    if tile_j > max_j:
                        max_j = tile_j
                
                mask_h = max_i + 1
                mask_w = max_j + 1
                shape_np_array = np.zeros((mask_h, mask_w), dtype=np.uint8)
                shape_view = shape_np_array
                for k in range(poly_ptr.mask.top // 2):
                    tile_i = poly_ptr.mask.data[k << 1]  # type: ignore
                    tile_j = poly_ptr.mask.data[(k << 1) + 1]  # type: ignore
                    shape_view[tile_i, tile_j] = 1  # type: ignore
                
                # Extract unoccupied spaces after placement
                unoccupied_spaces, space_sizes = extract_unoccupied_spaces(collage_array)
                # Create new collage metadata with updated state
                collage_meta.occupied_tiles = np.asarray(collage_array)
                collage_meta.unoccupied_spaces = unoccupied_spaces
                collage_meta.space_sizes = space_sizes
                collage_meta_ptr = <CollageMetadata*>malloc(sizeof(CollageMetadata))
                if collage_meta_ptr == NULL:
                    raise MemoryError("Failed to allocate CollageMetadata")
                collage_meta_ptr[0] = collage_meta  # type: ignore
                collages_pool.append(<object>collage_meta_ptr)
                # Create a new positions list for this collage with the first polyomino
                pos_struct.oy = current_poly.offset_i
                pos_struct.ox = current_poly.offset_j
                pos_struct.py = py
                pos_struct.px = px
                pos_struct.rotation = rotation
                pos_struct.frame = current_poly.frame
                pos_struct.shape = shape_np_array
                positions.append([_make_polyomino_position(pos_struct)])
                free(res)
    
    # Cleanup: free all IntStack masks in the sorted array and CollageMetadata pointers
    cdef int cleanup_idx
    for cleanup_idx in range(total_polyominos):
        IntStack_cleanup(&all_polyominoes[cleanup_idx].polyomino.mask)
    free(<void*>all_polyominoes)
    
    # Cleanup CollageMetadata pointers
    for j in range(len(collages_pool)):
        collage_meta_ptr = <CollageMetadata*>collages_pool[j]
        free(<void*>collage_meta_ptr)
    
    # Return all collages with their packed polyomino positions
    return positions


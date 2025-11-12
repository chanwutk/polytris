import typing

import numpy as np
from scipy import ndimage
from polyis.pack.cython.adapters import format_polyominoes


class PolyominoPosition(typing.NamedTuple):
    """Represents the position and orientation of a polyomino in a collage.
    
    Attributes:
        oy: Original y-coordinate offset of the polyomino from its video frame this polyomino belongs to
        ox: Original x-coordinate offset of the polyomino from its video frame this polyomino belongs to
        py: Packed y-coordinate position in the collage
        px: Packed x-coordinate position in the collage
        rotation: Rotation applied to the polyomino (0-3, representing 0°, 90°, 180°, 270°)
        frame: Frame index this polyomino belongs to
        shape: The actual polyomino shape as a numpy array
    """
    oy: int
    ox: int
    py: int
    px: int
    rotation: int
    frame: int
    shape: np.ndarray


class Placement(typing.NamedTuple):
    """Represents a successful placement of a polyomino in a collage.

    Attributes:
        y: Y-coordinate where the polyomino was placed
        x: X-coordinate where the polyomino was placed
        rotation: Rotation applied to the polyomino (0-3)
    """
    y: int
    x: int
    rotation: int


class CollageMetadata(typing.NamedTuple):
    """Holds a collage and cached metadata about its unoccupied regions.

    This NamedTuple maintains both the collage array and cached information about
    its unoccupied connected regions as polyomino masks to avoid expensive recomputation.

    Attributes:
        occupied_tiles: 2D numpy array representing the collage (0=empty, 1=occupied)
        unoccupied_spaces: List of 2D numpy arrays, each representing one unoccupied region
                          as a binary mask (same shape as occupied_tiles, 1=unoccupied in this region)
        space_sizes: List of integers, parallel to unoccupied_spaces, storing the size of each space
        largest_space: Integer representing the size of the largest unoccupied space in the collage
    """
    occupied_tiles: np.ndarray
    unoccupied_spaces: list[np.ndarray]
    space_sizes: list[int]
    largest_space: int


def extract_unoccupied_spaces(occupied_tiles: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    """Extract all unoccupied connected regions as polyomino masks and their sizes.

    Args:
        occupied_tiles: 2D numpy array representing the collage state

    Returns:
        Tuple of (list of binary masks, list of sizes)
        Each mask represents one unoccupied region (same shape as occupied_tiles, 1=unoccupied in this region, 0=otherwise)
        Each size is the number of tiles in the corresponding region
    """
    # Create binary mask where empty cells are 1
    empty_mask = (occupied_tiles == 0).astype(np.uint8)
    # Label connected components
    label_result = ndimage.label(empty_mask)
    # Assert type check: should be a tuple of (array, int)
    assert isinstance(label_result, tuple) and len(label_result) == 2, "ndimage.label should return a tuple of (labeled_array, num_features)"
    labeled_array, num_features = label_result
    assert isinstance(labeled_array, np.ndarray), "First element should be a numpy array"
    assert isinstance(num_features, (int, np.integer)), "Second element should be an integer"

    # Extract each region as a separate mask and calculate its size
    unoccupied_spaces = []
    space_sizes = []
    for region_id in range(1, num_features + 1):
        # Create a mask for this region (1 where region_id matches, 0 otherwise)
        region_mask = (labeled_array == region_id).astype(np.uint8)
        # Calculate size once when creating the mask
        region_size = np.sum(region_mask)
        unoccupied_spaces.append(region_mask)
        space_sizes.append(region_size)

    return unoccupied_spaces, space_sizes


def update_collage_after_placement(
    collage_meta: CollageMetadata,
    placement: Placement,
    polyomino_shape: np.ndarray
) -> CollageMetadata:
    """Update cached region information after a polyomino has been placed.

    Finds which unoccupied space contains the placement location and updates only that space.
    The placed polyomino breaks one unoccupied region into potentially smaller ones.

    Args:
        collage_meta: The CollageMetadata to update
        placement: Placement object containing the position where the polyomino was placed
        polyomino_shape: The shape of the polyomino that was placed

    Returns:
        New CollageMetadata with updated unoccupied spaces
    """
    # Get placement coordinates and dimensions
    py, px = placement.y, placement.x
    ph, pw = polyomino_shape.shape
    occupied_tiles = collage_meta.occupied_tiles
    
    # Find which unoccupied space contains the placement location
    # Find one tile that is part of the polyomino shape (guaranteed to be in exactly one unoccupied space)
    polyomino_tile_coords = np.where(polyomino_shape > 0)
    if len(polyomino_tile_coords[0]) == 0:
        raise ValueError("Polyomino has no tiles")
    
    # Get the first tile's coordinates relative to the polyomino shape
    tile_rel_y, tile_rel_x = polyomino_tile_coords[0][0], polyomino_tile_coords[1][0]
    # Convert to absolute coordinates in the collage
    tile_abs_y = py + tile_rel_y
    tile_abs_x = px + tile_rel_x
    
    # Find which unoccupied space contains this tile
    affected_space_idx = None
    for idx, space_mask in enumerate(collage_meta.unoccupied_spaces):
        # Check if this tile is in this unoccupied space
        if (tile_abs_y < space_mask.shape[0] and tile_abs_x < space_mask.shape[1] and
            space_mask[tile_abs_y, tile_abs_x] > 0):
            affected_space_idx = idx
            break
    
    # This should always succeed since the polyomino was successfully placed
    # If somehow it doesn't, fall back to full recalculation
    if affected_space_idx is None:
        raise ValueError("Failed to find the affected unoccupied space")
    
    # Get the affected unoccupied space mask
    affected_space = collage_meta.unoccupied_spaces[affected_space_idx]
    
    # Create a mask for the placed polyomino in the full collage coordinates
    placed_polyomino_mask = np.zeros_like(occupied_tiles, dtype=np.uint8)
    placed_polyomino_mask[py : py + ph, px : px + pw] = polyomino_shape
    
    # Subtract the placed polyomino from the affected space
    remaining_space = affected_space.copy()
    remaining_space = remaining_space.astype(np.int16) - placed_polyomino_mask.astype(np.int16)
    # Assert that the result is valid (all values are 0 or 1, since placed polyomino is a subset of the space)
    assert np.all((remaining_space >= 0) & (remaining_space <= 1)), "Remaining space contains invalid values after subtraction"
    # Convert to uint8 after assertion
    remaining_space = remaining_space.astype(np.uint8)
    
    # Extract new connected components from the remaining space
    # Label connected components in the remaining space
    label_result = ndimage.label(remaining_space)
    # Assert type check: should be a tuple of (array, int)
    assert isinstance(label_result, tuple) and len(label_result) == 2, "ndimage.label should return a tuple of (labeled_array, num_features)"
    assert isinstance(label_result[0], np.ndarray), "First element should be a numpy array"
    assert isinstance(label_result[1], (int, np.integer)), "Second element should be an integer"
    labeled_remaining, num_features = label_result
    
    # Create new unoccupied space masks for each connected component and calculate their sizes
    new_spaces = []
    new_space_sizes = []
    for region_id in range(1, num_features + 1):
        new_space_mask = (labeled_remaining == region_id).astype(np.uint8)
        # Calculate size once when creating the mask
        new_space_size = np.sum(new_space_mask)
        new_spaces.append(new_space_mask)
        new_space_sizes.append(new_space_size)
    
    # Build updated lists: keep unaffected spaces, replace affected space with new spaces
    updated_unoccupied_spaces = (
        collage_meta.unoccupied_spaces[:affected_space_idx] + 
        new_spaces + 
        collage_meta.unoccupied_spaces[affected_space_idx + 1:]
    )
    updated_space_sizes = (
        collage_meta.space_sizes[:affected_space_idx] + 
        new_space_sizes + 
        collage_meta.space_sizes[affected_space_idx + 1:]
    )
    
    # Calculate the largest space size from all updated space sizes
    # If there are no spaces, set to 0
    space_largest = max(updated_space_sizes) if updated_space_sizes else 0
    
    return CollageMetadata(occupied_tiles, updated_unoccupied_spaces, updated_space_sizes, space_largest)


def count_regions_at_least(collage_meta: CollageMetadata, min_size: int) -> int:
    """Count how many unoccupied regions are at least the given size.

    Args:
        collage_meta: The CollageMetadata to query
        min_size: Minimum size threshold

    Returns:
        Count of regions with size >= min_size
    """
    # Use cached space sizes for fast counting
    # Sum booleans directly (True=1, False=0) - faster than creating 1s
    return sum(size >= min_size for size in collage_meta.space_sizes)


def try_pack(polyomino: np.ndarray, occupied_tiles: np.ndarray) -> Placement | None:
    """Attempts to place a polyomino in a collage by trying all rotations and positions.
    
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
    # # Try all 4 possible rotations (0°, 90°, 180°, 270°)
    # for rotation in range(4):
    # Rotate the polyomino by 90 degrees * rotation counter-clockwise
    # rotated = np.rot90(polyomino, rotation)
    rotated = polyomino
    # Get dimensions of the rotated polyomino
    ph, pw = rotated.shape
    # Get dimensions of the collage
    ch, cw = occupied_tiles.shape

    # Try all possible positions where the polyomino would fit
    for y in range(ch - ph + 1):
        for x in range(cw - pw + 1):
            # Extract the window where we want to place the polyomino
            window = occupied_tiles[y : y + ph, x : x + pw]
            # Check if there's no overlap between existing collage and polyomino
            # Using bitwise AND: if any cell is occupied in both, placement is invalid
            if not np.any(window & rotated):
                # Place the polyomino by adding it to the collage
                occupied_tiles[y : y + ph, x : x + pw] += rotated
                # Return the successful placement coordinates and rotation
                # return Placement(y, x, rotation)
                return Placement(y, x, 0)


def pack_all(polyominoes_stacks: np.ndarray, h: int, w: int) -> list[list[PolyominoPosition]]:
    """Packs all polyominoes from multiple stacks into collages using a bin packing algorithm.

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
    # Initialize lists to store frame indices and polyomino data
    all_frames = []
    all_polyominoes: list[tuple[np.ndarray, tuple[int, int]]] = []

    # Combine stacks of polyominoes into a single list of polyominoes with frame indices
    for i in range(polyominoes_stacks.shape[0]):
        polyominoes_stack: np.uint64 = polyominoes_stacks[i]
        # Convert the memory address to a list of polyominoes (shapes and offsets)
        polyominoes = format_polyominoes(polyominoes_stack)
        # Record which frame each polyomino belongs to (frame index repeated for each polyomino)
        all_frames.extend([i] * len(polyominoes))
        # Add all polyominoes from this stack to the list
        all_polyominoes.extend(polyominoes)

    # Initialize storage for collages (with cached metadata) and their corresponding polyomino positions
    collages_pool: list[CollageMetadata] = []
    positions: list[list[PolyominoPosition]] = []

    # Combine polyominoes with their frame indices for processing
    all_polyominoes_frames = [*zip(all_polyominoes, all_frames)]
    # Sort polyominoes by size (largest first) for better packing efficiency
    all_polyominoes_frames.sort(key=lambda x: np.sum(x[0][0]), reverse=True)
    
    # Process each polyomino in size order (largest first)
    for polyomino, frame in all_polyominoes_frames:
        # Extract the shape and original offset coordinates
        shape, (oy, ox) = polyomino
        
        # Calculate the size of the polyomino (number of tiles)
        polyomino_size = np.sum(shape)

        # Try to place the polyomino in an existing collage
        # Evaluate each collage by the largest unoccupied space size
        collage_candidates = []
        for i, collage_meta in enumerate(collages_pool):
            # First, check if there's enough total empty space
            empty_space = np.sum(collage_meta.occupied_tiles == 0)
            if empty_space >= polyomino_size:
                # Use cached largest space information to evaluate collage quality
                # Check if the largest space can fit the polyomino
                largest_space = collage_meta.largest_space
                if largest_space >= polyomino_size:
                    collage_candidates.append((i, largest_space))

        # Sort by largest unoccupied space size (descending order)
        # Collages with smaller largest contiguous empty spaces are prioritized
        collage_candidates.sort(key=lambda x: x[1], reverse=False)

        # Try to pack in collages with the smallest largest unoccupied space first
        for i, _ in collage_candidates:
            collage_meta = collages_pool[i]
            # Attempt to pack the polyomino in this collage
            res = try_pack(shape, collage_meta.occupied_tiles)
            if res is not None:
                # Successfully placed - extract position and rotation
                py, px, rotation = res
                # Record the polyomino position in this collage
                positions[i].append(PolyominoPosition(oy, ox, py, px, rotation, frame, shape))
                # Update cached region information for this collage only
                collages_pool[i] = update_collage_after_placement(collage_meta, res, shape)
                # Move to next polyomino (break out of collage loop)
                break
        else:
            # No existing collage could fit this polyomino - create a new collage
            # Create a new empty collage with specified dimensions
            collage_array = np.zeros((h, w), dtype=np.uint8)
            # Attempt to place the polyomino in the new collage
            res = try_pack(shape, collage_array)
            # This should always succeed since the collage is empty
            assert res is not None
            # Extract position and rotation from successful placement
            py, px, rotation = res
            # Extract unoccupied spaces after placement
            unoccupied_spaces, space_sizes = extract_unoccupied_spaces(collage_array)
            # Calculate the largest space size from all space sizes
            # If there are no spaces, set to 0
            space_largest = max(space_sizes) if space_sizes else 0
            # Create new collage metadata with updated state
            collages_pool.append(CollageMetadata(collage_array, unoccupied_spaces, space_sizes, space_largest))
            # Create a new positions list for this collage with the first polyomino
            positions.append([PolyominoPosition(oy, ox, py, px, rotation, frame, shape)])
    
    # Return all collages with their packed polyomino positions
    return positions

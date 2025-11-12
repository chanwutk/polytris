import typing

import numpy as np
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

    # Initialize storage for collages and their corresponding polyomino positions
    collages_pool: list[np.ndarray] = []
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
        # Calculate empty space for each collage and filter/sort by most empty space
        collage_candidates = []
        for i, collage in enumerate(collages_pool):
            empty_space = np.sum(collage == 0)
            # Filter out collages with less empty space than the polyomino size
            if empty_space >= polyomino_size:
                collage_candidates.append((i, empty_space))

        # Sort by most empty space first (descending order)
        collage_candidates.sort(key=lambda x: x[1], reverse=True)

        # Try to pack in the collages with most empty space first
        for i, _ in collage_candidates:
            collage = collages_pool[i]
            # Attempt to pack the polyomino in this collage
            res = try_pack(shape, collage)
            if res is not None:
                # Successfully placed - extract position and rotation
                py, px, rotation = res
                # Record the polyomino position in this collage
                positions[i].append(PolyominoPosition(oy, ox, py, px, rotation, frame, shape))
                # Move to next polyomino (break out of collage loop)
                break
        else:
            # No existing collage could fit this polyomino - create a new collage
            # Create a new empty collage with specified dimensions
            collage = np.zeros((h, w), dtype=np.uint8)
            # Add the new collage to the pool
            collages_pool.append(collage)
            # Attempt to place the polyomino in the new collage
            res = try_pack(shape, collage)
            # This should always succeed since the collage is empty
            assert res is not None
            # Extract position and rotation from successful placement
            py, px, rotation = res
            # Create a new positions list for this collage with the first polyomino
            positions.append([PolyominoPosition(oy, ox, py, px, rotation, frame, shape)])
    
    # Return all collages with their packed polyomino positions
    return positions

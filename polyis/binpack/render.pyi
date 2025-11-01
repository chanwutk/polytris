"""Type stubs for render module."""

import typing


if typing.TYPE_CHECKING:
    from polyis.dtypes import NPImage, PolyominoPositions


def render(
    canvas: NPImage,
    positions: list[PolyominoPositions],
    frame: NPImage,
    chunk_size: int,
):
    """
    Render packed polyominoes onto the canvas.
    
    Args:
        canvas: The canvas to render onto
        positions: List of packed polyominoe positions
        frame: Source frame
        chunk_size: Size of each tile/chunk
    """
    ...
import numpy as np

def group_tiles(bitmap_input: np.ndarray, tilepadding_mode: int) -> list[tuple[np.ndarray, tuple[int, int]]]:
    ...

def format_polyominoes(polyomino_stack_ptr: int) -> list[tuple[np.ndarray, tuple[int, int]]]:
    ...

def pack_append(
    polyominoes: list[tuple[np.ndarray, tuple[int, int]]],
    h: int,
    w: int,
    occupied_tiles: np.ndarray
) -> list[tuple[int, int, np.ndarray, tuple[int, int]]] | None:
    ...

def get_polyominoes(polyominoes: list[tuple[np.ndarray, tuple[int, int]]]) -> int:
    ...

def format_positions(positions: list) -> list[tuple[int, int, np.ndarray, tuple[int, int]]] | None:
    ...
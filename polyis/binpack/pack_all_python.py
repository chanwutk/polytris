import typing

import numpy as np
from polyis.binpack.adapters import format_polyominoes


class PolyominoPosition(typing.NamedTuple):
    oy: int
    ox: int
    py: int
    px: int
    rotation: int
    frame: int
    shape: np.ndarray


class Placement(typing.NamedTuple):
    y: int
    x: int
    rotation: int


def try_pack(polyomino: np.ndarray, collage: np.ndarray) -> Placement | None:
    for rotation in range(4):
        rotated = np.rot90(polyomino, rotation)
        ph, pw = rotated.shape
        ch, cw = collage.shape

        for y in range(ch - ph + 1):
            for x in range(cw - pw + 1):
                window = collage[y : y + ph, x : x + pw]
                if not np.any(window & rotated):
                    collage[y : y + ph, x : x + pw] += rotated
                    return Placement(y, x, rotation)


def pack_all(polyominoes_stacks: list[int], h: int, w: int) -> list[list[PolyominoPosition]]:
    all_frames = []
    all_polyominoes: list[tuple[np.ndarray, tuple[int, int]]] = []

    for i, polyominoes_stack in enumerate[int](polyominoes_stacks):
        polyominoes = format_polyominoes(polyominoes_stack)
        all_frames.extend([i] * len(polyominoes))
        all_polyominoes.extend(polyominoes)

    collages_pool: list[np.ndarray] = []
    positions: list[list[PolyominoPosition]] = []

    all_polyominoes_frames = [*zip(all_polyominoes, all_frames)]
    all_polyominoes_frames.sort(key=lambda x: -np.sum(x[0][0]))
    for polyomino, frame in all_polyominoes_frames:
        shape, (oy, ox) = polyomino
        for i, collage in enumerate(collages_pool):
            res = try_pack(shape, collage)
            if res is not None:
                py, px, rotation = res
                positions[i].append(PolyominoPosition(oy, ox, py, px, rotation, frame, shape))
                break
        else:
            collage = np.zeros((h, w), dtype=np.uint8)
            collages_pool.append(collage)
            res = try_pack(shape, collage)
            assert res is not None
            py, px, rotation = res
            positions.append([PolyominoPosition(oy, ox, py, px, rotation, frame, shape)])
    return positions

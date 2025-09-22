import queue
from typing import Literal, TypeAlias, TypeGuard, TypeVar, TypeVarTuple, Unpack

import numpy as np
import numpy.typing as npt


S1 = Literal[1]
S2 = Literal[2]
S3 = Literal[3]
S4 = Literal[4]
S5 = Literal[5]
S6 = Literal[6]
S7 = Literal[7]
S8 = Literal[8]
S9 = Literal[9]
S10 = Literal[10]
S11 = Literal[11]
S12 = Literal[12]
S13 = Literal[13]
S14 = Literal[14]
S15 = Literal[15]
S16 = Literal[16]

NPDType = TypeVar("NPDType", bound=np.generic)
D1 = tuple[int]
D2 = tuple[int, int]
D3 = tuple[int, int, int]
D4 = tuple[int, int, int, int]
D5 = tuple[int, int, int, int, int]
D6 = tuple[int, int, int, int, int, int]
D7 = tuple[int, int, int, int, int, int, int]
D8 = tuple[int, int, int, int, int, int, int, int]

SType = TypeVarTuple("SType", default=Unpack[tuple[int, ...]])
Array: TypeAlias = np.ndarray[tuple[Unpack[SType]], np.dtype[NPDType]]  # type: ignore

BArray: TypeAlias = Array[tuple[Unpack[SType]], np.bool]
IArray: TypeAlias = Array[tuple[Unpack[SType]], np.integer]
FArray: TypeAlias = Array[tuple[Unpack[SType]], np.floating]

NPImage = Array[int, int, Literal[3], np.uint8]
IdPolyominoOffset = tuple[int, Array[*D2, np.bool], tuple[int, int]]

PipeDatum = TypeVar("PipeDatum")
Pipe: TypeAlias = queue.Queue[PipeDatum | None]

InPipe = Pipe
OutPipe = Pipe

DetArray = Array[int, S5, np.number]
IntDetArray = IArray[int, S5]
FloatDetArray = FArray[int, S5]


def is_det_array(x: npt.NDArray) -> TypeGuard[DetArray]:
    return x.ndim == 2 and x.shape[1] == 5 and np.issubdtype(x.dtype, np.number) and x.dtype != np.bool_

def is_np_image(x: npt.NDArray) -> TypeGuard[NPImage]:
    return x.ndim == 3 and x.shape[2] == 3 and x.dtype == np.uint8

Bitmap2D = Array[*D2, np.uint8]

def is_bitmap(x: np.ndarray[tuple, np.dtype[NPDType]]) -> TypeGuard[Bitmap2D]:
    return x.ndim == 2

IndexMap = Array[tuple[int, int], np.uint16]

def is_index_map(x: np.ndarray[tuple, np.dtype[NPDType]]) -> TypeGuard[IndexMap]:
    return x.ndim == 2 and x.dtype == np.uint16

Polyomino = tuple[Bitmap2D, tuple[int, int]]
PolyominoPositions = tuple[int, int, Bitmap2D, tuple[int, int]]
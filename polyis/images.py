from typing import TypeGuard, NewType

import cv2
import torch


ImgCHW = NewType('ImgCHW', torch.Tensor)
ImgHWC = NewType('ImgHWC', torch.Tensor)

ImgNCHW = NewType('ImgNCHW', torch.Tensor)
ImgNHWC = NewType('ImgNHWC', torch.Tensor)

ImgGCHW = NewType('ImgGCHW', torch.Tensor)
ImgGHWC = NewType('ImgGHWC', torch.Tensor)


def isCHW(img: torch.Tensor) -> TypeGuard[ImgCHW]:
    return img.dim() == 3 and img.shape[0] in {1, 3}

def isHWC(img: torch.Tensor) -> TypeGuard[ImgHWC]:
    return img.dim() == 3 and img.shape[2] in {1, 3}

def isNCHW(img: torch.Tensor) -> TypeGuard[ImgNCHW]:
    return img.dim() == 4 and img.shape[1] in {1, 3}

def isNHWC(img: torch.Tensor) -> TypeGuard[ImgNHWC]:
    return img.dim() == 4 and img.shape[3] in {1, 3}

def isGCHW(img: torch.Tensor) -> TypeGuard[ImgGCHW]:
    return img.dim() == 5 and img.shape[2] in {1, 3}

def isGHWC(img: torch.Tensor) -> TypeGuard[ImgGHWC]:
    return img.dim() == 5 and img.shape[4] in {1, 3}


def padCHW(img: ImgCHW, h: int, w: int) -> ImgCHW:
    C, H, W = img.shape
    pad_h = (h - H % h) % h
    pad_w = (w - W % w) % w
    ret = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
    assert isCHW(ret), ret.shape
    return ret

def padHWC(img: ImgHWC, h: int, w: int) -> ImgHWC:
    H, W, C = img.shape
    pad_h = (h - H % h) % h
    pad_w = (w - W % w) % w
    ret = torch.nn.functional.pad(img, (0, 0, 0, pad_w, 0, pad_h))
    assert isHWC(ret), ret.shape
    return ret


def splitCHW(img: ImgCHW, h: int, w: int) -> ImgGCHW:
    assert img.shape[1] % h == 0
    assert img.shape[2] % w == 0
    ret = (img
        .unfold(1, h, h)
        .unfold(2, w, w)
        .permute(1, 2, 0, 3, 4)
    )
    assert isGCHW(ret), ret.shape
    return ret

def splitHWC(img: ImgHWC, h: int, w: int) -> ImgGHWC:
    assert img.shape[0] % h == 0
    assert img.shape[1] % w == 0
    ret = (img
        .unfold(0, h, h)
        .unfold(1, w, w)
        .permute(0, 1, 3, 4, 2)
    )
    assert isGHWC(ret), ret.shape
    return ret


def reconstructGCHW(img: ImgGCHW) -> ImgCHW:
    _img = img.permute(2, 0, 3, 1, 4)
    C, GH, H, GW, W = _img.shape
    ret = _img.reshape(C, GH * H, GW * W)
    assert isCHW(ret), ret.shape
    return ret

def reconstructGHWC(img: ImgGHWC) -> ImgHWC:
    _img = img.permute(0, 2, 1, 3, 4)
    GH, H, GW, W, C = _img.shape
    ret = _img.reshape(GH * H, GW * W, C)
    assert isHWC(ret), ret.shape
    return ret


def flatGCHW(img: ImgGCHW) -> ImgNCHW:
    GH, GW, *DIM = img.shape
    ret = img.reshape(GH * GW, *DIM)
    assert isNCHW(ret), ret.shape
    return ret

def flatGHWC(img: ImgGHWC) -> ImgNHWC:
    GH, GW, *DIM = img.shape
    ret = img.reshape(GH * GW, *DIM)
    assert isNHWC(ret), ret.shape
    return ret


def foldNCHW(img: ImgNCHW, h: int, w: int) -> ImgGCHW:
    N, *DIM = img.shape
    assert N == h * w
    ret = img.reshape(h, w, *DIM)
    assert isGCHW(ret), ret.shape
    return ret

def foldNHWC(img: ImgNHWC, h: int, w: int) -> ImgGHWC:
    N, *DIM = img.shape
    assert N == h * w
    ret = img.reshape(h, w, *DIM)
    assert isGHWC(ret), ret.shape
    return ret


if __name__ == '__main__':
    cap = cv2.VideoCapture('videos/jnc00.mp4')
    ret, frame = cap.read()
    cap.release()
    frame = frame[32 * 10:32 * (10 + 17), 64 * 3: 64 * (3 + 19)]
    # cv2.imwrite('frame.jpg', frame)

    tframe = torch.tensor(frame)
    assert isHWC(tframe), tframe.shape
    img_ghwc = splitHWC(tframe, 32, 64)
    for i in range(17):
        for j in range(19):
            # cv2.imwrite(f'frame_{i}_{j}.jpg', img_ghwc[i, j].numpy())
            assert torch.all(img_ghwc[i, j] == torch.tensor(frame[i * 32:(i + 1) * 32, j * 64:(j + 1) * 64]))


    img = torch.tensor([*range(3 * 5 * 7 * 11 * 13)]).reshape(3, 5 * 13, 7 * 11)
    assert isCHW(img), img.shape
    img_gchw = splitCHW(img, 13, 11)
    for i in range(5):
        for j in range(7):
            assert torch.all(img_gchw[i, j] == img[:, i * 13:(i + 1) * 13, j * 11:(j + 1) * 11])
    
    assert torch.all(reconstructGCHW(img_gchw) == img)

    img_nchw = flatGCHW(img_gchw)
    for i in range(5):
        for j in range(7):
            assert torch.all(img_nchw[(i * 7) + j] == img_gchw[i, j])
    
    assert torch.all(foldNCHW(img_nchw, 5, 7) == img_gchw)


    img = torch.tensor([*range(3 * 5 * 7 * 11 * 13)]).reshape(5 * 13, 7 * 11, 3)
    assert isHWC(img), img.shape
    img_ghwc = splitHWC(img, 13, 11)
    for i in range(5):
        for j in range(7):
            assert torch.all(img_ghwc[i, j] == img[i * 13:(i + 1) * 13, j * 11:(j + 1) * 11])
    
    assert torch.all(reconstructGHWC(img_ghwc) == img)

    img_nhwc = flatGHWC(img_ghwc)
    for i in range(5):
        for j in range(7):
            assert torch.all(img_nhwc[(i * 7) + j] == img_ghwc[i, j])
    
    assert torch.all(foldNHWC(img_nhwc, 5, 7) == img_ghwc)
    
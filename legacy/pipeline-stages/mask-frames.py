import sys, pathlib
MODULES_PATH = pathlib.Path().absolute() / 'modules'
sys.path.append(str(MODULES_PATH))


import json
import os
import shutil
from xml.etree import ElementTree

import cv2
from matplotlib.path import Path
import numpy as np
import torch

import polyis.images


CONFIG = './modules/b3d/configs/config_refined.json'
DIR = './videos'
MASK_DIR = os.path.join('pipeline-stages', 'video-masked')


def get_bitmap(width: int, height: int, mask: ElementTree.Element):
    domain = mask.find('.//polygon[@label="domain"]')
    assert domain is not None

    domain = domain.attrib['points']
    domain = domain.replace(';', ',')
    domain = np.array([
        float(pt) for pt in domain.split(',')]).reshape((-1, 2))
    tl = (int(np.min(domain[:, 1])), int(np.min(domain[:, 0])))
    br = (int(np.max(domain[:, 1])), int(np.max(domain[:, 0])))
    domain_poly = Path(domain)
    # width, height = int(frame.shape[1]), int(frame.shape[0])
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T
    bitmap = domain_poly.contains_points(pixel_points)
    bitmap = bitmap.reshape((height, width, 1))

    bitmap = bitmap[tl[0]:br[0], tl[1]:br[1], :]
    return bitmap, tl, br


def logger(queue):
    with open('crop.jsonl', 'w') as fc:
        while True:
            val = queue.get()
            if val == None:
                return
            fc.write(val + '\n')
            fc.flush()


def process(gpuIdx: int, file: str, mask, logger_queue):

    SAMPLE_SIZE = 1

    cap = cv2.VideoCapture(os.path.join(DIR, file))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask_img = mask.find(f'.//image[@name="{file.split(".")[0]}.jpg"]')
    assert mask_img is not None

    bitmask, mtl, mbr = get_bitmap(width, height, mask_img)
    logger_queue.put(json.dumps({'tl': mtl, 'br': mbr, 'file': file}))
    bitmask = torch.from_numpy(bitmask).to(f'cuda:{gpuIdx}').to(torch.bool)

    width, height = mbr[1] - mtl[1], mbr[0] - mtl[0]
    # logger_queue.put(f'Processing {file} with shape {width}x{height}')
    writer = cv2.VideoWriter(os.path.join(MASK_DIR, file), cv2.VideoWriter.fourcc(*'mp4v'), int(30 / SAMPLE_SIZE), (width, height))

    idx = -1
    while cap.isOpened():
        # if idx > 5000:
        #     break
        idx += 1
        print(idx)
        success, frame = cap.read()
        if not success:
            break

        # if idx > 500:
        #     break
        # if idx % SAMPLE_SIZE != 0:
        #     continue

        # mask_frame(frame, mask)
        frame = torch.from_numpy(frame).to(f'cuda:{gpuIdx}')[mtl[0]:mbr[0], mtl[1]:mbr[1], :] * bitmask
        assert polyis.images.isHWC(frame), frame.shape

        assert (height, width) == frame.shape[:2], (height, width, frame.shape[:2])
        height, width = frame.shape[:2]

        # pad so that the width and height are multiples of CHUNK_SIZE
        # frame = polyis.images.padHWC(frame, CHUNK_SIZE, CHUNK_SIZE)

        frame = frame.detach().cpu().numpy() #.transpose((1, 2, 0))
        frame = frame.astype(np.uint8)

        writer.write(frame)
    writer.release()
    cap.release()


def main():
    if os.path.exists(MASK_DIR):
        shutil.rmtree(MASK_DIR)
    os.mkdir(MASK_DIR)

    fc = open('crop.jsonl', 'w')

    tree = ElementTree.parse('./pipeline-stages/masks.xml')
    mask = tree.getroot()

    num_cuda = torch.cuda.device_count()

    logger_queue = torch.multiprocessing.Queue()
    logger_process = torch.multiprocessing.Process(target=logger, args=(logger_queue,))
    logger_process.start()

    ps = []
    for fidx, file in enumerate(os.listdir(DIR)):
        p = torch.multiprocessing.Process(target=process, args=(fidx % num_cuda, file, mask, logger_queue))
        p.start()
        ps.append(p)
        # cap = cv2.VideoCapture(os.path.join(DIR, file))
        # width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # mask_img = mask.find(f'.//image[@name="{file.split(".")[0]}.jpg"]')
        # assert mask_img is not None

        # bitmask, mtl, mbr = get_bitmap(width, height, mask_img)
        # fc.write(json.dumps({'tl': mtl, 'br': mbr}) + '\n')
        # bitmask = torch.from_numpy(bitmask).to('cuda:1').to(torch.bool)

        # width, height = mbr[1] - mtl[1], mbr[0] - mtl[0]
        # writer = cv2.VideoWriter(f'./video-masked/{file}', cv2.VideoWriter.fourcc(*'mp4v'), 15, (width, height))

        # idx = -1
        # while cap.isOpened():
        #     idx += 1
        #     print(idx)
        #     success, frame = cap.read()
        #     if not success:
        #         break

        #     if idx % 2 == 0:
        #         continue

        #     # mask_frame(frame, mask)
        #     frame = torch.from_numpy(frame).to('cuda:1') * bitmask
        #     assert polyis.images.isHWC(frame), frame.shape

        #     height, width = frame.shape[:2]

        #     # pad so that the width and height are multiples of CHUNK_SIZE
        #     frame = polyis.images.padHWC(frame, CHUNK_SIZE, CHUNK_SIZE)

        #     frame = frame.detach().cpu().numpy()#.transpose((1, 2, 0))
        #     frame = frame.astype(np.uint8)

        #     writer.write(frame)
        # writer.release()
        # cap.release()
    for p in ps:
        p.join()
        p.terminate()
    logger_queue.put(None)
    logger_process.join()
    logger_process.terminate()
    cv2.destroyAllWindows()
    fc.close()


if __name__ == '__main__':
    main()
import os
import multiprocessing as mp

import cv2


DIR = '../videos'


def combine_frames(filename: str):
    frames = None
    cap = cv2.VideoCapture(filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = frame_count // 500
    count = 0
    while True:
        if count * step >= frame_count:
            break
        print(filename, count * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, count * step)
        ret, frame = cap.read()
        if not ret:
            break
        if frames is None:
            frames = frame.astype('uint64')
        else:
            frames += frame.astype('uint64')
        count += 1
    cap.release()

    assert frames is not None
    frame = (frames / count).astype('uint8')
    # save image
    cv2.imwrite(filename.replace('.mp4', '.jpg'), frame)


def main():
    processes: list[mp.Process] = []
    for file in os.listdir(DIR):
        if file.endswith('.mp4'):
            video_path = os.path.join(DIR, file)
            p = mp.Process(target=combine_frames, args=(video_path,))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
        p.terminate()


if __name__ == '__main__':
    main()
    print('done')
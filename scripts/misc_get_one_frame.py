#!/usr/local/bin/python

import os

import cv2

for file in os.listdir('videos'):
    if file.endswith('.mp4'):
        video_path = os.path.join('videos', file)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join('output', file.replace('.mp4', '.jpg')), frame)
        cap.release()
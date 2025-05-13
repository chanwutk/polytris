import sys
import os

sys.path.append('/data/chanwutk/projects/minivan/modules/darknet')

import json
import numpy
import os
import shutil

import cv2
import sys

cwd = os.getcwd()
os.chdir('/data/chanwutk/projects/minivan/modules/darknet')
import darknet
os.chdir(cwd)



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Example detection and tracking script')
    parser.add_argument('-v', '--videos', required=True, help='Input video files / directory')
    parser.add_argument('-o', '--output', default='/data/chanwutk/projects/minivan/det', help='Path to output detections')
    return parser.parse_args()


def main(args):
    videofiles = args.videos
    output = args.output

    if os.path.isdir(videofiles):
        videofiles = [os.path.join(videofiles, f) for f in os.listdir(videofiles) if f.endswith('.mp4')]
    else:
        videofiles = [videofiles]
    
    data_root = '/data/chanwutk/data/otif-dataset'
    batch_size = 1
    width = 704
    height = 480
    param_width = 704
    param_height = 480
    threshold = 0.25
    classes = ''
    label = 'caldot'


    def eprint(s):
        sys.stderr.write(str(s) + "\n")
        sys.stderr.flush()

    if classes != '':
        classes = {cls.strip(): True for cls in classes.split(',')}
    else:
        classes = None

    detector_label = label
    if detector_label.startswith('caldot'):
        detector_label = 'caldot'
    if detector_label in ['amsterdam', 'jackson']:
        detector_label = 'generic'

    config_path = os.path.join(data_root, 'yolov3', detector_label, 'yolov3-{}x{}-test.cfg'.format(param_width, param_height))
    meta_path = os.path.join(data_root, 'yolov3', detector_label, 'obj.data')
    names_path = os.path.join(data_root, 'yolov3', detector_label, 'obj.names')

    if detector_label == 'generic':
        weight_path = os.path.join(data_root, 'yolov3', detector_label, 'yolov3.best')
    else:
        weight_path = os.path.join(data_root, 'yolov3', detector_label, 'yolov3-{}x{}.best'.format(param_width, param_height))

    # ensure width/height in config file
    with open(config_path, 'r') as f:
        tmp_config_buf = ''
        for line in f.readlines():
            line = line.strip()
            if line.startswith('width='):
                line = 'width={}'.format(width)
            if line.startswith('height='):
                line = 'height={}'.format(height)
            tmp_config_buf += line + "\n"
    tmp_config_path = '/tmp/yolov3-{}.cfg'.format(os.getpid())
    with open(tmp_config_path, 'w') as f:
        f.write(tmp_config_buf)

    # Write out our own obj.data which has direct path to obj.names.
    tmp_obj_names = '/tmp/obj-{}.names'.format(os.getpid())
    shutil.copy(names_path, tmp_obj_names)

    with open(meta_path, 'r') as f:
        tmp_meta_buf = ''
        for line in f.readlines():
            line = line.strip()
            if line.startswith('names='):
                line = 'names={}'.format(tmp_obj_names)
            tmp_meta_buf += line + "\n"
    tmp_obj_meta = '/tmp/obj-{}.data'.format(os.getpid())
    with open(tmp_obj_meta, 'w') as f:
        f.write(tmp_meta_buf)

    # Finally we can load YOLOv3.
    net, class_names, _ = darknet.load_network(tmp_config_path, tmp_obj_meta, weight_path, batch_size=batch_size)
    os.remove(tmp_config_path)
    os.remove(tmp_obj_names)
    os.remove(tmp_obj_meta)



    for videofile in sorted(videofiles, key=lambda x: int(os.path.basename(x).split('.')[0])):
        if int(os.path.basename(videofile).split('.')[0]) < 14:
            continue
        cap = cv2.VideoCapture(videofile)
        # writer = cv2.VideoWriter(os.path.join(output, os.path.basename(videofile)), cv2.VideoWriter.fourcc(*'mp4v'), 30, (param_width, param_height))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fidx = 0
        with open(os.path.join(output, os.path.basename(videofile) + '.jsonl'), 'w') as f:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # frame = frame[120:, 50:500]
                frame = cv2.resize(frame, dsize=[param_width, param_height])
                height, width = frame.shape[:2]
                arr = numpy.array([frame], dtype='uint8')

                arr = arr.transpose((0, 3, 1, 2))
                arr = numpy.ascontiguousarray(arr.flat, dtype='float32')/255.0
                darknet_images = arr.ctypes.data_as(darknet.POINTER(darknet.c_float))
                darknet_images = darknet.IMAGE(width, height, 3, darknet_images)
                raw_detections = darknet.network_predict_batch(net, darknet_images, batch_size, width, height, threshold, 0.5, None, 0, 0)
                detections = []
                for idx in range(batch_size):
                    num = raw_detections[idx].num
                    raw_dlist = raw_detections[idx].dets
                    darknet.do_nms_obj(raw_dlist, num, len(class_names), 0.45)
                    raw_dlist = darknet.remove_negatives(raw_dlist, class_names, num)
                    dlist = []
                    for cls, score, (cx, cy, w, h) in raw_dlist:
                        if classes is not None and cls not in classes:
                            continue
                        dlist.append({
                            'class': cls,
                            'score': float(score),
                            'left': int(cx-w/2),
                            'right': int(cx+w/2),
                            'top': int(cy-h/2),
                            'bottom': int(cy+h/2),
                        })
                    detections.append(dlist)
                darknet.free_batch_detections(raw_detections, batch_size)
                print('json'+json.dumps(detections), flush=True)
                f.write(json.dumps([fidx, [[d['left'], d['top'], d['right'], d['bottom']] for d in detections[0]]]) + '\n')
                # f.flush()

                # frame0 = frame0.astype('uint8')
                # for det in detections[0]:
                #     x1 = det['left']
                #     y1 = det['top']
                #     x2 = det['right']
                #     y2 = det['bottom']

                #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # frame = numpy.ascontiguousarray(frame, dtype='uint8')
                # writer.write(frame)
                fidx += 1
            cap.release()
            # writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    main(args)
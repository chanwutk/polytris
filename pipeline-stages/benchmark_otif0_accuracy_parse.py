import os
import json
import shutil

OTIF_RES_DIR = '/data/chanwutk/projects/otif/pipeline2/outputs/caldot1/ours-simple'
OTIF_OUT_DIR = '/data/chanwutk/projects/polyis/det_otif'

if os.path.exists(OTIF_OUT_DIR):
    shutil.rmtree(OTIF_OUT_DIR)
os.makedirs(OTIF_OUT_DIR)

for d in os.listdir(OTIF_RES_DIR):
    # assert os.isdir(os.path.join(OTIF_RES_DIR, d)), d
    if os.path.exists(os.path.join(OTIF_OUT_DIR, d)):
        shutil.rmtree(os.path.join(OTIF_OUT_DIR, d))
    os.makedirs(os.path.join(OTIF_OUT_DIR, d))
    for f in os.listdir(os.path.join(OTIF_RES_DIR, d)):
        with open(os.path.join(OTIF_RES_DIR, d, f), 'r') as fi:
            dets = json.load(fi)
            if dets is None:
                dets = []
        with open(os.path.join(OTIF_OUT_DIR, d, f+'l'), 'w') as file:
            file.write(json.dumps([0, []]) + '\n')
            for i, det in enumerate(dets):
                # dets[i] = [det[0], det[1], det[2], det[3], det[4], 0, 0, 0, 0, 0]
                fdet = []
                if det is not None:
                    for dd in det:
                        fdet.append([dd['track_id'], dd['left'], dd['top'], dd['right'], dd['bottom']])
                
                file.write(json.dumps([i + 1, fdet]) + '\n')


from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
from PIL import Image
import requests
import time

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('./split.jpg')

image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small")
# model = model.to('cuda')

# print(image_processor)
# print(model)

proc = []
infe = []
post = []
# for i in range(5):
for i in range(100):
    start = time.time()
    inputs = image_processor(images=image, return_tensors="pt")
    end = time.time()
    proc.append(end - start)


    start = time.time()
    outputs = model(**inputs)
    end = time.time()
    infe.append(end - start)

    # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    start = time.time()
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        0
    ]
    end = time.time()
    post.append(end - start)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

print(sum(proc), sum(infe), sum(post))

import numpy as np
runtime = (np.array(model.runtime).sum(axis=0))
print((runtime * 100 / sum(runtime)).astype(int))

runtime = ((np.array(model.vit.runtime).sum(axis=0)))
print((runtime))
print((runtime * 100 / sum(runtime)).astype(int))

for layer in model.vit.encoder.layer:
    runtime_dict = {}
    # print(layer.runtime)
    for rr in layer.runtime:
        for key, r in rr:
            if key not in runtime_dict:
                runtime_dict[key] = []
            runtime_dict[key].append(r) # type: ignore
    
    for key, rs in runtime_dict.items():
        print(key)
        print(sum(rs))
    print()
    print()
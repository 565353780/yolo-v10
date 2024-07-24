import sys
sys.path.append('../yolov10')

import os
import json
import torch
from tqdm import trange

from ultralytics import YOLOv10

model_file_path = '../yolov10/runs/detect/v5-yolov10x-train4k-no_mosiac-no_erase-final/weights/best.pt'
test_image_folder_path = '/home/chli/Dataset/X-Ray/test1/'
save_folder_name = 'yolov10x-train4k-thr01'
conf_threshold = 0.01

save_folder_path = './output/' + save_folder_name
os.makedirs(save_folder_path + '/', exist_ok=True)

model = YOLOv10(model_file_path, 'detect')

image_filename_list = os.listdir(test_image_folder_path)

image_filename_list.sort()

results_list = []

with torch.no_grad():
    for i in trange(len(image_filename_list)):
        image_filename = image_filename_list[i]
        if image_filename[-4:] != '.jpg':
            continue

        current_results = []

        image_file_path = test_image_folder_path + image_filename

        result = model.predict(image_file_path, conf=conf_threshold)[0]
        result.save(filename=save_folder_path + '/' + str(i) + '.jpg')

        boxes = result.boxes.data.detach().clone().cpu().numpy()

        for i in range(8):
            current_label_boxes = boxes[boxes[:, 5].astype(int) == i]

            current_results.append(current_label_boxes[:, :5].tolist())

        results_list.append(current_results)

    with open(save_folder_path + '.json', 'w') as f:
        json.dump(results_list, f)

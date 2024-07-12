import sys
sys.path.append('../yolov10')

import os
import json
import torch
from tqdm import tqdm

from ultralytics import YOLOv10

with torch.no_grad():
    model_file_path = '../yolov10/runs/detect/train/weights/best.pt'
    test_image_folder_path = '/home/chli/Dataset/X-Ray/test1/'

    model = YOLOv10(model_file_path, 'detect')

    image_filename_list = os.listdir(test_image_folder_path)

    image_filename_list.sort()

    results_list = []

    for image_filename in tqdm(image_filename_list):
        if image_filename[-4:] != '.jpg':
            continue

        current_results = []

        image_file_path = test_image_folder_path + image_filename

        results = model.predict(image_file_path)[0]
        boxes = results.boxes.data.detach().clone().cpu().numpy()

        for i in range(8):
            current_label_boxes = boxes[boxes[:, 5].astype(int) == i]

            current_results.append(current_label_boxes[:, :5].tolist())

        results_list.append(current_results)

    os.makedirs('./output/', exist_ok=True)

    with open('./output/xray_yolov10_all.json', 'w') as f:
        json.dump(results_list, f)

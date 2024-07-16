import sys
sys.path.append('../yolov10')

from ultralytics import YOLOv10

model_yaml_folder_path = './yolo_v_ten/Config/Model/'
train_yaml_file_path = './yolo_v_ten/Config/Train/xray.yaml'
dataset_yaml_file_path = './yolo_v_ten/Config/Dataset/xray.yaml'

model = YOLOv10(model_yaml_folder_path + 'xray-v1.yaml', 'detect')
# model = YOLOv10('../yolov10/runs/detect/v4-yolov10x-train4k-final/weights/best.pt')

model.train(cfg=train_yaml_file_path, data=dataset_yaml_file_path)

import sys
sys.path.append('../yolov10')

from ultralytics import YOLOv10

model_yaml_file_path = './yolo_v_ten/Config/Model/yolov10n.yaml'
train_yaml_file_path = './yolo_v_ten/Config/Train/xray.yaml'
dataset_yaml_file_path = './yolo_v_ten/Config/Dataset/xray.yaml'

# model = YOLOv10(model_yaml_file_path, 'detect')
model = YOLOv10('../yolov10/runs/detect/train2/weights/last.pt')

model.train(cfg=train_yaml_file_path, data=dataset_yaml_file_path)

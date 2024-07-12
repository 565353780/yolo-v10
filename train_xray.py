import sys
sys.path.append('../yolov10')

from ultralytics import YOLOv10

model_yaml_file_path = './yolo_v_ten/Config/Model/yolov10x.yaml'

dataset_yaml_file_path = './yolo_v_ten/Config/xray.yaml'

model = YOLOv10(model_yaml_file_path, 'detect')

model.train(data=dataset_yaml_file_path, epochs=500, batch=16, imgsz=640)

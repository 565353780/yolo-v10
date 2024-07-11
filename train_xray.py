import sys
sys.path.append('../yolov10')

from ultralytics import YOLOv10

pretrained_model_file_path = '/home/chli/chLi/Model/yolov10/yolov10x.pt'

yaml_file_path = './yolo_v_ten/Config/xray.yaml'

model = YOLOv10(pretrained_model_file_path, 'detect')

# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')

# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.train(data=yaml_file_path, epochs=500, batch=256, imgsz=640)

cd ..
git clone https://github.com/THU-MIG/yolov10.git

pip install -U torch torchvision torchaudio

pip install -U onnx onnxruntime pycocotools pyyaml scipy \
  onnxsim gradio opencv-python psutil py-cpuinfo \
  huggingface-hub safetensors seaborn tensorboard

if [ "$(uname)" = "Linux" ]; then
  pip install -U onnxruntime-gpu
fi

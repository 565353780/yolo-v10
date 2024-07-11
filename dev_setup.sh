pip install -U torch torchvision torchaudio

pip install -U onnx onnxruntime pycocotools pyyaml scipy \
  onnxsim gradio opencv-python psutil py-cpuinfo \
  huggingface-hub safetensors

if [ "$(uname)" = "Linux" ]; then
  pip install -U onnxruntime-gpu
fi

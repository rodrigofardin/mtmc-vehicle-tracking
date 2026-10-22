# Imagem única: pipeline MTMC, API de streaming, testes e benchmark de detectores (TensorRT FP16).
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 YOLO_CONFIG_DIR=/tmp/ultralytics     HF_HOME=/workspace/models/cache/hf TORCH_HOME=/workspace/models/cache/torch PYTHONPATH=/workspace
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git     && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt
# treino de Re-ID (reid_training/): converte o ONNX do OSNet-AIN para PyTorch
RUN pip install onnx2torch

# libs CUDA do pip (cuDNN/cuBLAS/...) e do TensorRT no caminho de carga (onnxruntime-gpu e TensorRT as reutilizam)
ENV NV=/opt/conda/lib/python3.11/site-packages/nvidia
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/tensorrt_libs:${NV}/cudnn/lib:${NV}/cublas/lib:${NV}/cufft/lib:${NV}/curand/lib:${NV}/cusparse/lib:${NV}/cusolver/lib:${NV}/cuda_runtime/lib:${NV}/cuda_nvrtc/lib:${NV}/nvjitlink/lib:/opt/conda/lib/python3.11/site-packages/torch/lib:/opt/conda/lib

# Código, configs e testes vão na imagem; dataset, modelos e saídas são montados (ver docker-compose.yml)
COPY mtmc ./mtmc
COPY configs ./configs
COPY tests ./tests

ENTRYPOINT ["python", "-m", "mtmc"]
CMD ["--help"]

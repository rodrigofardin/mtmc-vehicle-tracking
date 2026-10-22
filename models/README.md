# Modelos (não versionados)

Os pesos não são commitados (tamanho e licenças próprias). Coloque aqui:

| Arquivo | Uso | Onde obter |
|---|---|---|
| `yolov8n.pt`, `yolov8l.pt`, `yolov9e.pt`, `yolo12x.pt` | detectores (`detector.model`) | assets do Ultralytics (baixados automaticamente pelo benchmark: `python -m mtmc.detector_bench_cli`) |
| `osnet_ain_x1_0_vehicle_reid.onnx` | Re-ID de veículos (`reid.model`) | `https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx` — SHA-384 `0515ce72f653c39780d5b87dfed7255d396dd2b1e8b6e91fbaacdfad1da189166343157273c02f3b0fede3050ef7abb7` (Open Model Zoo `vehicle-reid-0001`) |
| `engines/*.engine` | engines TensorRT FP16 | gerados automaticamente a partir dos pesos (dependem da GPU e da versão do TensorRT) |
| RF-DETR / D-FINE | detectores DETR | baixados no primeiro uso (`rfdetr` e Hugging Face `ustc-community/dfine-*-coco`) |

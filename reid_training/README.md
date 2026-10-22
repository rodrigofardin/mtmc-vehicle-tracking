# Treino de Re-ID de veículos (ajuste fino com o GT do CityFlowV2)

Ajusta o OSNet-AIN (`vehicle-reid-0001`, pré-treinado em VeRi) com os veículos anotados do CityFlowV2 e exporta um ONNX
pronto para o pipeline (`reid.model`). Tudo roda na imagem Docker do projeto (GPU).

## Protocolo (evita vazamento)
- **Treino:** S03, S04 (train) e S05 (validation): 426 veículos, câmeras e cenas diferentes.
- **Avaliação reservada:** S01 e S02 (240 veículos). Os IDs não se repetem entre cenários e esses cenários **nunca** entram no
  treino (`build_dataset.py` recusa S01/S02 sem `--eval-set`). A época final é a que vale: não há seleção de checkpoint pelos dados
  de avaliação.
- Os recortes vêm das caixas do GT (mín. 48 px, sem corte pela borda), até 25 por (câmera, veículo) no treino e 10 na avaliação.

## Passo a passo (a partir da raiz do repositório)
```bash
# 1) conjuntos de recortes (o dataset é montado em /AICity22_Track1_MTMC_Tracking; ver docker-compose.yml)
docker compose run --rm --entrypoint python mtmc -m reid_training.build_dataset --root /AICity22_Track1_MTMC_Tracking \
    --scenes train/S03 train/S04 validation/S05 --out reid_training/data/train --per-track 25
docker compose run --rm --entrypoint python mtmc -m reid_training.build_dataset --root /AICity22_Track1_MTMC_Tracking \
    --eval-set --scenes train/S01 validation/S02 --out reid_training/data/eval --per-track 10

# 2) treino (≈ 2 min/época numa RTX 2060; 40 épocas)
docker compose run --rm --entrypoint python mtmc -m reid_training.train --out reid_training/runs/ft1

# 3) exporta para ONNX (compara com o PyTorch) e avalia nos cenários reservados
docker compose run --rm --entrypoint python mtmc -m reid_training.export_onnx --ckpt reid_training/runs/ft1/last.pt --embed bn --out models/osnet_ain_vehicle_reid_ft.onnx
docker compose run --rm --entrypoint python mtmc -m reid_training.evaluate --onnx models/osnet_ain_vehicle_reid_ft.onnx
```

## Método
- Backbone: ONNX do OSNet-AIN convertido para PyTorch com `onnx2torch` (paridade numérica verificada: diferença ≈ 8e-7; o
  conversor exigiu um ajuste para o BatchNorm final, em `model.load_backbone`). 2,19 M de parâmetros.
- Cabeça: BNNeck (BatchNorm1d 512) + classificador de IDs (só no treino).
- Perdas: entropia cruzada com label smoothing (0,1) + triplete batch-hard (margem 0,3) em embeddings normalizados.
- Amostragem P×K = 16×4, com recortes de 2 câmeras por veículo (positivos entre câmeras).
- Aumentos: recorte aleatório (simula caixas de detector), flip, brilho/contraste, random erasing.
- AdamW (backbone 1e-4, cabeça 1e-3), aquecimento de 3 épocas + cosseno, AMP FP16, 40 épocas × 150 iterações.

## Resultados
Veja [RESULTS.md](RESULTS.md): rank-1 no S01 0,677 → 0,835 e no S02 0,635 → 0,814; IDF1 do pipeline +4 pontos nos dois cenários. Dados, recortes e pesos **não** são
versionados (licença do dataset).

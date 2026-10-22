# MTMC — Multi-Camera Multi-Object Tracking (veículos)

Pipeline modular para rastrear veículos entre várias câmeras: alinhamento temporal de vídeos, detecção + ByteTrack por
câmera, tracklets, **Re-ID de veículos**, associação global com evidências, avaliação contra ground truth, vídeos de
diagnóstico e um **monitor web em tempo real**. Desenvolvido e medido no **CityFlowV2 (AI City 2022, Track 1)**.

> **Estado honesto:** funciona de ponta a ponta (offline e ao vivo, em GPU com TensorRT), mas a qualidade da associação
> global ainda é modesta: IDF1 global ≈ 0,61 no S01 (cenário onde os limiares foram ajustados) e ≈ 0,38 no S02 (cenário
> não usado no ajuste). Veja [Resultados](#resultados) e [Limitações](#limitações).

```
vídeos ─► gerenciador/alinhamento temporal ─► detector (PyTorch | TensorRT FP16) ─► ByteTrack (1 por câmera)
        ─► tracklets ─► Re-ID (OSNet-AIN, ONNX) + homografia ─► associação global (com evidências) ─► vídeos / métricas
                                         └─► API FastAPI (WebSocket binário) ─► frontend React (nginx)
```

## Conteúdo
- `mtmc/` — pacote Python (CLI `python -m mtmc ...`): `videos`, `metadata`, `detectors`, `detector_export*`, `local_tracker`,
  `detect_track`, `tracklets`, `reid`, `calibration`, `association`, `live`, `live_assoc`, `server`, `evaluation`,
  `reid_check`, `detector_bench*`, `visualize`.
- `frontend/` — monitor web (React + TypeScript + Vite + Tailwind + shadcn/ui), serviço Docker independente
  ([frontend/README.md](frontend/README.md)).
- `configs/` — YAML (`s01_cityflow.yaml`, `s02_cityflow.yaml`, `live_trt_s01.yaml`, `exp_*.yaml`, `detector_bench.yaml`).
- `tests/` — 74 testes do backend (sintéticos, sem depender dos vídeos). `outputs/results/` — resultados medidos (JSON).

## Requisitos
Docker com GPU NVIDIA (testado: RTX 2060 6 GB, driver CUDA 13, Docker Desktop/WSL2). A imagem (`Dockerfile`) usa
PyTorch 2.7 + CUDA 12.6, Ultralytics 8.4, TensorRT 10, onnxruntime-gpu, RF-DETR e `transformers`.
**Dataset:** [AI City Challenge 2022 Track 1 (CityFlowV2)](https://www.aicitychallenge.org/) — não é redistribuído aqui
(licença própria); baixe e aponte `AICITY_DIR`. **Pesos:** veja [models/README.md](models/README.md).

## Início rápido
```bash
docker compose build mtmc                       # imagem única (CLI, API, testes, benchmark)
docker compose run --rm tests                   # 74 testes
# pesos: coloque yolov8n.pt/yolov8l.pt e o ONNX de Re-ID em models/ (ver models/README.md)
docker compose run --rm mtmc run -c configs/s01_cityflow.yaml --no-viz        # pipeline completo no S01 + métricas vs GT
docker compose up -d --build                    # API ao vivo + frontend (http://localhost:8080)
MTMC_CONFIG=configs/live_trt_s01.yaml docker compose up -d mtmc-api   # ao vivo com detector TensorRT
```
`AICITY_DIR` (padrão `D:/AICity22_Track1_MTMC_Tracking`), `FRONTEND_PORT` (8080) e `API_PORT` (8000) são variáveis do compose.
Comandos da CLI: `inspect`, `detect`, `tracklets`, `associate`, `evaluate`, `visualize`, `run`, `serve`, `reid-check`, `tune`,
`snapshot`. Sem GPU/Docker há um modo CPU (venv com PyTorch CPU; com `torch 2.4.0+cpu` no Windows é preciso
`detector.disable_mkldnn: true`, pois o oneDNN produziu saídas absurdas e não determinísticas).

## Dados do CityFlowV2 e semântica verificada
| Item | Achado |
|---|---|
| `cam_timestamp` | instantes **iniciais** dos vídeos em segundos (ReadMe.txt item 13). Vídeos podem ter frames pulados/travados: offset constante é aproximação |
| `cam_framenum` | **total** de frames (conferido por decodificação nas 5 câmeras do S01) |
| `calibration.txt` | homografia: `inv(H)` leva pixel → (lat, lon) (verificado contra o centro GPS do ReadMe); erro de reprojeção 5–11 px |
| GT (`gt/gt.txt`) | MOTChallenge com ID global, **só veículos vistos em ≥ 2 câmeras** |
| `cam_loc/*.png` | mapa ilustrativo sem escala/coordenadas; não usado na associação |

`t_global = frame/fps + offset_câmera`. Frames com o mesmo índice **não** são tratados como simultâneos.

## Resultados
Todos os números abaixo foram medidos aqui (arquivos em [`outputs/results/`](outputs/results/)). IDF1 global é implementação
própria (ROI + só IDs em ≥ 2 câmeras, IoU ≥ 0,5; **não** é o `eval.py` oficial); HOTA não foi calculado.

### Re-ID (OSNet-AIN `vehicle-reid-0001`, ONNX, GPU)
Em 1197 recortes do GT (95 veículos, 5 câmeras do S01): entrada RGB; AUC 0,951; rank-1 0,687; mAP 0,513 (BGR: 0,924 / 0,594).
Cosseno médio: mesmo veículo 0,42; veículos diferentes 0,10 (p90 0,24) ⇒ limiares de aparência ≈ 0,15. Treinado em VeRi
(diferença de domínio).

### Pipeline completo (detector YOLOv8n, limiares ajustados no S01)
| Cenário | IDF1 | IDP | IDR | precisão/recall de pares |
|---|---|---|---|---|
| S01 (ajustado nele) | 0,635 | 0,59 | 0,69 | 93 % / 51 % |
| S02 (não usado no ajuste) | 0,39 | 0,34 | 0,46 | 81 % / 48 % |

O cue espacial (homografia, 80 m) soma ≈ +3 pontos de IDF1 no S01 (sem ele: 0,60). O S02 tem só 112 tracklets com GT.

### Detectores maiores + TensorRT FP16 (RTX 2060)
Benchmark em 1717 frames do GT (S01+S02), frame 1080p completo incluindo pré/pós-processamento, entrada 544×960
(DETRs: 704² / 640²). Recall = IoU ≥ 0,5, conf ≥ 0,25, classe-agnóstico. AP50 parcial só serve para ranquear (GT incompleto).

| Modelo (backend) | recall | AP50 parcial | caixas/frame | latência p50 | fps | VRAM |
|---|---|---|---|---|---|---|
| YOLOv8n (PyTorch FP16) | 0,867 | 0,577 | 9,0 | 24,8 ms | 40 | 74 MB |
| YOLOv8n (TensorRT FP16) | 0,870 | 0,567 | 9,1 | 17,4 ms | 58 | 84 MB |
| YOLOv8l (TRT) | 0,893 | 0,606 | 10,5 | 20,9 ms | 48 | 206 MB |
| YOLOv9-E (TRT) | 0,885 | 0,593 | 10,5 | 25,4 ms | 39 | 490 MB |
| YOLOv10-L (TRT) | 0,885 | 0,596 | 9,9 | 21,4 ms | 47 | 194 MB |
| YOLOv10-X (TRT) | 0,890 | 0,596 | 10,2 | 22,5 ms | 44 | 212 MB |
| YOLOv12-X (TRT) | 0,886 | 0,611 | 10,0 | 32,0 ms | 31 | 344 MB |
| RF-DETR-M (TRT) | 0,880 | 0,200* | 13,3 | 28,3 ms | 35 | 480 MB |
| RF-DETR-L (TRT) | 0,882 | 0,164* | 13,9 | 22,5 ms | 44 | 372 MB |
| D-FINE-L (TRT) | 0,867 | 0,176* | 29,3 | 25,9 ms | 39 | 142 MB |
| D-FINE-X (TRT) | 0,888 | 0,233* | 28,4 | 26,7 ms | 38 | 284 MB |

\* os DETRs devolvem muitas caixas de baixa confiança em veículos não anotados; o AP parcial os penaliza. O YOLOv12-L com os
pesos originais não carrega no Ultralytics (`AAttn` sem `qkv`): omitido. As latências são dominadas pelo pré-processamento
em CPU (frame 1080p), não pela GPU. A exportação FP16 usa nosso builder (`detector_export.py`) porque o caminho
`format="engine"` do Ultralytics 8.4 exige `nvidia-modelopt`.

**Impacto no pipeline completo** (mesmos Re-ID, limiares e tracker; só o detector muda):

| Detector | S01 IDF1 | S01 IDR | S01 trocas de ID locais | S02 IDF1 | S02 IDR | S02 recall de pares |
|---|---|---|---|---|---|---|
| YOLOv8n | **0,613** | 0,650 | 49 | **0,383** | 0,457 | 0,45 |
| YOLOv8l | 0,591 | 0,695 | 17 | 0,330 | 0,451 | 0,50 |
| YOLOv9-E | 0,547 | 0,696 | 13 | 0,325 | 0,441 | 0,71 |
| YOLOv12-X | 0,584 | 0,703 | 9 | 0,356 | 0,486 | 0,71 |
| RF-DETR-L | 0,556 | 0,705 | 22 | 0,318 | 0,452 | 0,85 |
| D-FINE-X (conf 0,25 / 0,4 / 0,5) | 0,392 / 0,424 / 0,463 | 0,52–0,58 | 1262 / 889 / 618 | — | — | — |

**Leitura:** detectores maiores **não** melhoram o IDF1 global neste GT; eles aumentam o recall de identidades (IDR),
estabilizam o tracking local (menos trocas de ID) e melhoram a associação entre câmeras (recall de pares 0,71–0,85 no S02 contra
0,45), mas detectam também veículos **não anotados** (o GT só cobre veículos vistos em ≥ 2 câmeras), o que reduz a precisão
(IDP) e, portanto, o IDF1. D-FINE precisa de ajuste do tracker (muitas caixas instáveis por frame). Limiares de associação não
foram reajustados por detector.

### Ao vivo (5 câmeras, RTX 2060)
YOLOv8-L TensorRT FP16 + ByteTrack + Re-ID + associação ao vivo: ≈ 8 fps processados por câmera (≈ 40 fps no total; o restante
dos frames é descartado para acompanhar o tempo real), rodada de associação ≈ 1,8 s. Antes (YOLOv8n PyTorch, 960 px): ≈ 5 fps e
5,6 s. A inferência de GPU é serializada por um lock (várias threads inicializando engines TensorRT juntas causavam
`CUDA error ... during capture`). A qualidade ao vivo não foi medida contra GT.

## Associação global
Para cada par de tracklets de câmeras diferentes: relação temporal (simultâneo/transição), compatibilidade de classe e cues
decisivos — **aparência** (cosseno do Re-ID) e **espacial** (homografia p/ plano em metros, ou zonas/transições configuradas).
Cue disponível e reprovado ⇒ rejeita; sem cue decisivo ⇒ `insufficient_evidence`; candidatos de score próximo ⇒ `ambiguous`;
nunca duas observações da mesma câmera na mesma identidade. Cada decisão e evidência fica em
`association_decisions.jsonl` / `global_identities.json`. Ao vivo (`server.live_association`), `mtmc/live_assoc.py` reexecuta a
mesma lógica a cada ~2 s com IDs estáveis; só veículos ligados entre ≥ 2 câmeras recebem `gid` (podem mudar entre rodadas).

## Configuração
Tudo em YAML (`extends:` herda outro arquivo). Principais blocos: `cameras`, `metadata`, `timestamp`/`framenum`, `detector`
(`type: ultralytics_track | ultralytics | rfdetr | dfine`, `trt: true`, `imgsz`, `conf`, `classes`), `tracker`, `reid`
(`model`, `preprocess`, `input_size`), `spatial` (`calibration`, `zones`, `transitions`, `max_ground_dist`), `association`
(`appearance_threshold`, `appearance_scale`, `accept_score`, ...), `evaluation` (`gt_dir`, `roi_pattern`), `tuning.grid`,
`server` (`live_association`, ...). Adicionar/remover câmeras é só editar `cameras:`.

## Testes
`docker compose run --rm tests` → 74 passam (leitura de metadados, alinhamento temporal, múltiplas câmeras, tracking local
independente, associação correta/incorreta/ambígua/sem Re-ID, avaliação, API de streaming, associador ao vivo, ByteTrack,
calibração). Frontend: 41 testes Vitest + 6 Playwright (ver `frontend/README.md`). Os testes de detecção/tracking usam um
YOLO falso para o fluxo; TensorRT, RF-DETR, D-FINE e o Re-ID real só são exercitados pelos benchmarks/execuções reais.

## Limitações
- IDF1 global modesto e sensível ao cenário (S01 ≈ 0,61–0,64 ajustado; S02 ≈ 0,32–0,39). Limiares ajustados no S01.
- IDF1 próprio (não oficial); HOTA ausente; GT parcial subestima a precisão de detectores mais sensíveis.
- Re-ID treinado em VeRi (não em CityFlow); calibração com erro de 5–11 px; frames pulados nos vídeos.
- Ao vivo: ≈ 8 fps/câmera na RTX 2060 (não chega a 10 fps em 5 câmeras); qualidade não medida com GT; `gid` pode mudar.
- D-FINE e YOLOv12-L (pesos originais) não ficaram utilizáveis nesta configuração. Sem autenticação/TLS no serviço web.

## Licença e créditos
Código sob **AGPL-3.0** ([LICENSE](LICENSE)), compatível com o uso do Ultralytics (AGPL-3.0). Terceiros: Ultralytics
(YOLOv8/9/10/12, ByteTrack), RF-DETR (Roboflow, Apache-2.0 até o L), D-FINE, OSNet-AIN / Open Model Zoo `vehicle-reid-0001`
(Intel; base MIT/Apache), CityFlowV2 / AI City Challenge (dataset com licença própria, não redistribuído).

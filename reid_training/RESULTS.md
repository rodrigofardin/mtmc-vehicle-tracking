# Resultados do ajuste fino do Re-ID

Treino: S03, S04, S05 (426 veículos, 49 238 recortes). Avaliação: S01 e S02, **nunca vistos no treino**. Modelo final = época 40
(sem seleção de época nos dados de avaliação). 40 épocas × 150 iterações, P×K = 16×4, ≈ 97 min numa RTX 2060.
Dados brutos em `outputs/results/` (`reid_eval_*.json`, `reid_ft_history.json`).

## Recuperação entre câmeras (recortes do GT, ONNX avaliado com onnxruntime)
| Cenário | Modelo | rank-1 | mAP | AUC | cosseno mesmo veículo (médio / p10) | cosseno veículos diferentes (médio / p90) |
|---|---|---|---|---|---|---|
| S01 | original (OSNet-AIN / VeRi) | 0,677 | 0,453 | 0,934 | 0,407 / 0,193 | 0,101 / 0,248 |
| S01 | **ajustado** | **0,835** | **0,683** | **0,978** | 0,549 / 0,344 | 0,145 / 0,283 |
| S02 | original | 0,635 | 0,504 | 0,963 | 0,426 / 0,240 | 0,091 / 0,228 |
| S02 | **ajustado** | **0,814** | **0,723** | **0,986** | 0,559 / 0,370 | 0,114 / 0,254 |

Acompanhamento durante o treino (embedding sem BN): rank-1 S01 0,813 → 0,826 → 0,833 → 0,833 nas épocas 10/20/30/40; ganho
quase todo nas primeiras 10 épocas. O treino chega a ≈ 100 % de acurácia de classificação nos veículos de treino (sobreajuste ao
treino), mas o desempenho nos cenários reservados não piora. A exportação ONNX difere do PyTorch em 1,6e-5.

## Pipeline completo (detector YOLOv8n TensorRT; só o Re-ID e os limiares de associação mudam)
Limiares do modelo ajustado escolhidos no **S01** (varredura de 24 combinações em `outputs/results/tuning_ft_s01.json`:
aparência 0,25, score 0,5, espaço 150 m; platô: 0,647–0,653 para limiares de 0,20 a 0,30) e aplicados sem mudança no **S02**.

| Cenário | Re-ID | IDF1 | IDP | IDR | precisão / recall de pares |
|---|---|---|---|---|---|
| S01 | original | 0,613 | 0,581 | 0,650 | 0,93 / 0,47 |
| S01 | **ajustado** | **0,652** | 0,593 | 0,725 | 0,92 / 0,56 |
| S02 | original | 0,383 | 0,329 | 0,457 | 0,83 / 0,45 |
| S02 | **ajustado** | **0,424** | 0,368 | 0,500 | 1,00 / 0,50 |

Ganho de ≈ +4 pontos de IDF1 nos dois cenários, bem menor que o ganho no Re-ID isolado (rank-1 +16 a +18 pontos): o IDF1
global também depende do tracker local, da calibração e da sincronização. IDF1 é implementação própria (não o `eval.py`
oficial) e o S02 tem poucos tracklets com GT (112), então a precisão de pares 1,00 não deve ser lida como perfeita.
O ajuste fino melhora sobretudo o **recall de identidades** (IDR); a precisão (IDP) sobe pouco.

## Limitações
- Mesmo domínio de câmeras/cidade do CityFlowV2: não foi testado em dados de outra origem.
- Os cenários de avaliação são os mesmos usados para ajustar os limiares de associação (S01) e para reportar o teste (S02).
- Caixas do GT são mais "soltas" que as do detector; o treino usa recortes aleatórios para simular isso, sem garantia.
- Os pesos treinados não são versionados (licença do dataset).

# Diagnóstico da associação: onde o IDF1 se perde

Ferramenta: `docker compose run --rm mtmc diagnose -c configs/exp_ft_s01_yolov8n.yaml` (código em `mtmc/diagnose.py`; resultados completos em
`outputs/results/diagnosis_s01.json` e `diagnosis_s02.json`). Configuração: YOLOv8n TensorRT + Re-ID ajustado, limiares do experimento
`exp_ft_*` (aparência 0,25, score 0,5, espaço 150 m). IDF1 é implementação própria (ROI + só IDs em ≥ 2 câmeras).

## Experimentos oráculo (troca uma etapa por uma versão perfeita)
| Experimento | S01 IDF1 | S01 recall de pares | S02 IDF1 | S02 recall de pares |
|---|---|---|---|---|
| A. pipeline real | 0,652 | 0,557 | 0,424 | 0,500 |
| B. associação perfeita (ID do GT por tracklet) | 0,753 | 1,00 | 0,386 | 1,00 |
| C. Re-ID perfeito + regras reais | 0,658 | 0,563 | 0,344 | 0,568 |
| C2. Re-ID perfeito, sem homografia | 0,674 | 0,599 | 0,344 | 0,568 |
| D. Re-ID real, sem homografia | 0,644 | 0,489 | 0,405 | 0,500 |

Limitação dos oráculos B/C: o rótulo do tracklet é o veículo do GT que casa em ≥ 50% das suas detecções; tracklets que não passam desse corte
(veículos não anotados ou caixas ruins) ficam sem rótulo. No S02, 459 dos 572 tracklets ficam sem rótulo, por isso B/C ficam *abaixo* de A (não são
teto estrito).

**Leitura (S01).** Um Re-ID perfeito, com as regras atuais, **não melhora** o pipeline (0,652 → 0,658). O Re-ID deixou de ser o gargalo; o limite está nas regras de
associação e nos tracklets. Associação perfeita sobre os tracklets atuais chegaria a 0,753: esse é o teto imposto por detector + tracker local.

## Por que os pares verdadeiros não são ligados (S01: 137 de 309 pares perdidos; S02: 22 de 44)
| Causa | S01 | S02 |
|---|---|---|
| ligado (direto ou por cadeia) | 172 | 22 |
| classes incompatíveis (ex.: car × truck, regra estrita) | **50** | **11** |
| tracklet curto demais (< 10 observações) | 35 | 5 |
| aceito pelos cues mas bloqueado (ambíguo/conflito) | 24 | 1 |
| intervalo ou sobreposição fora da janela temporal | 12 | 3 |
| reprovado só pelo espaço / só pela aparência | 6 / 5 | 0 / 0 |
| score combinado abaixo do mínimo | 5 | 2 |

- **A regra de classe estrita é a maior causa isolada** (16% dos pares verdadeiros no S01, 25% no S02): a mesma caminhonete/SUV vira `car` numa câmera e `truck` em outra.
- Falsos links: 14 no S01 (7 em transições, 4 simultâneos).
- Os cues, isolados, discriminam bem (S01): aparência AUC 0,975; espaço AUC 0,92. **Combinados em pares** (cosseno ≥ 0,40 **e** distância ≤ 20 m) dão F1 0,87
  (precisão 0,95, recall 0,80), contra precisão 0,92 / recall 0,56 do pipeline. Ou seja, há margem na regra de decisão. (Os positivos ficam a 4 m de mediana, p90 = 24 m;
  os negativos, a 44 m de mediana.) Limite: é um ótimo em pares; o IDF1 depende também do agrupamento, e a varredura anterior preferiu espaço frouxo (150 m).

## Cobertura do GT pelos tracks (teto de detector + tracker local)
| | S01 | S02 |
|---|---|---|
| caixas do GT casadas por uma detecção rastreada | 84% | **63%** |
| fração de (câmera, veículo) com cobertura < 50% | 21% | **58%** |
| tracklets por (câmera, veículo) (fragmentação) | 1,08 | 1,08 |
| tracklets com rótulo de GT | 273 / 510 | 113 / 572 |

No S02 a perda está **antes da associação**: caixas de altura média (60–150 px) são cobertas em apenas ~34% (c006: 1038/3094; c009: 1456/4279), enquanto caixas grandes (> 150 px)
chegam a ~86–91%. O ByteTrack só **inicia** trilhas com confiança ≥ 0,6 (`new_track_thresh`), e 41% das detecções rastreadas têm confiança < 0,6; o YOLOv8n a 544×960 enxerga pior
veículos médios em 1080p. A fragmentação em tracklets *não* é o problema (1,08 por veículo/câmera).

## O que isso indica
1. **Regras de associação** (S01): relaxar a classe estrita (grupos car/truck/bus ou penalização suave), aceitar tracklets mais curtos e tratar ambiguidade/conflito com agrupamento
   global em vez de bloqueio por pares. Potencial por pares: recall 0,56 → ~0,80.
2. **Detecção/tracker** (S02): limiares do tracker (`new_track_thresh`, `track_high_thresh`) e resolução/modelo para veículos médios; ROI antes do tracker.
3. Re-ID: prioridade baixa agora (Re-ID perfeito não muda o IDF1 com as regras atuais).

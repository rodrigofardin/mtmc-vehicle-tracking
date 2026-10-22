# Melhoria das regras de associação (pós-diagnóstico)

Motivação: o [diagnóstico](DIAGNOSTICO.md) mostrou que um Re-ID perfeito não melhorava o IDF1 com as regras atuais; as maiores perdas eram a **classe estrita**
(16% dos pares verdadeiros no S01), tracklets curtos e o bloqueio por ambiguidade/conflito na fusão gulosa. Detector e tracker ficaram fixos
(YOLOv8n TensorRT, ByteTrack, Re-ID ajustado, detecções reaproveitadas); só a associação mudou. IDF1 é implementação própria (ROI + só IDs em ≥ 2 câmeras).

## 1. Regras: o que a varredura (S01) mostrou
Varredura de 432 combinações (`outputs/results/tuning_rules_s01.json`), média do IDF1 por fator:
- **Classe:** estrita 0,632 → grupo `[car, truck, bus]` **0,656** (+2,4 pontos). `[car, truck]` rende igual.
- Tracklets mais curtos (mín. 10 → 6 → 4 observações): 0,651 → 0,647 → 0,647 (sem ganho; piora um pouco).
- Margem de ambiguidade 0,10 → 0,03: 0,650 → 0,647 (sem efeito relevante).
Melhor configuração no S01: grupos de classe + aparência 0,20 + score 0,5 + espaço 150 m ⇒ **IDF1 0,675** (era 0,652), precisão de pares 0,96, recall 0,68.

## 2. Pontuação aprendida + agrupamento global (`association.method: learned`)
- `mtmc/pair_scorer.py`: dois modelos logísticos (pares simultâneos × transições) com features de aparência (cosseno e cosseno²), classe igual (deixa de ser veto),
  comprimento/confiança e **geometria no plano de chão** (distância média e p90, direção e velocidade comparadas nos simultâneos; intervalo, distância saída→entrada,
  velocidade implícita e coerência de direção nas transições). Prior corrigido para probabilidades calibradas. Treino: `train-scorer` (rótulos do GT; nada de GT na inferência).
- `mtmc/cluster.py`: agrupamento hierárquico de ligação média com restrições: ligação média ≥ `cluster_threshold` (logit), **veto** se algum par conhecido é claramente
  diferente (`veto_logit`) e no máximo uma observação por câmera por identidade. Substitui a fusão gulosa por pares e o bloqueio por ambiguidade.
- Mesma saída de antes (identidades + decisões com evidências: logit, p, features), portanto avaliação, vídeos e o associador ao vivo funcionam sem mudanças.

Validação por veículo (2 dobras, `train-scorer`): simultâneos AUC 0,985/0,999 (F1 em p=0,5: 0,93/0,95); transições AUC 0,977/0,980 (F1 0,65/0,77).

## 3. Resultados (testes em cena que o modelo não viu)
Limiares do agrupamento (τ = −1, veto = −5) escolhidos no S01 (em amostra; `tuning_learned_s01_insample.json`: platô 0,684–0,685 para τ ∈ {−2, −1}).

| Cenário | Método | IDF1 | IDP | IDR | precisão / recall de pares |
|---|---|---|---|---|---|
| S02 | regras originais | 0,424 | 0,368 | 0,500 | 1,00 / 0,50 |
| S02 | regras melhoradas (escolhidas no S01) | 0,437 | 0,369 | 0,537 | 1,00 / 0,64 |
| S02 | **aprendida (treinada no S01)** | **0,458** | 0,415 | 0,510 | 1,00 / 0,55 |
| S01 | regras originais | 0,652 | 0,593 | 0,725 | 0,92 / 0,56 |
| S01 | regras melhoradas (em amostra) | 0,675 | 0,612 | 0,752 | 0,96 / 0,68 |
| S01 | aprendida em amostra (treinada no S01; otimista) | 0,685 | 0,620 | 0,766 | 0,99 / 0,70 |
| S01 | aprendida (treinada no S02, só 36 pares positivos) | 0,621 | 0,619 | 0,623 | 0,99 / 0,58 |

- **Ganho limpo no S02: 0,424 → 0,458 (+3,4 pontos)** com a pontuação aprendida; as regras melhoradas dão +1,3. A sensibilidade é baixa: IDF1 de 0,445 a 0,458 para regularização
  0,1–100 e τ entre −2 e −1 (todos acima do baseline).
- O modelo treinado no S02 (muito pouco dado) **piora** o S01 (0,621 < 0,652): a pontuação aprendida depende de dados de treino suficientes.
- O teto com associação perfeita sobre os mesmos tracklets era 0,753 no S01; o S02 está limitado antes, pela cobertura de detecção (63%).

## Limitações
- Há só duas cenas com tracklets gerados pelo pipeline completo; o único teste limpo da pontuação aprendida é S01 → S02. Treinar com S03/S04/S05 exigiria embeddings de um Re-ID que não viu esses
  veículos (o Re-ID ajustado foi treinado neles, o que infla os cossenos de treino).
- Não validado no modo ao vivo (tracklets parciais e fragmentados): o padrão continua `method: rules`; para experimentar, defina `association.method: learned` e `association.scorer`.
- Limiares do agrupamento escolhidos em amostra no S01.

## Como reproduzir
```bash
docker compose run --rm mtmc train-scorer -c configs/exp_ft_s01_yolov8n.yaml --scorer-out models/pair_scorer_s01.json
docker compose run --rm mtmc associate -c configs/exp_learned_s02.yaml && docker compose run --rm mtmc evaluate -c configs/exp_learned_s02.yaml
docker compose run --rm mtmc tune -c configs/exp_learned_s01_insample.yaml --grid '{"association.cluster_threshold":[-3,-2,-1,0,1],"association.veto_logit":[-8,-5,-3]}'
```

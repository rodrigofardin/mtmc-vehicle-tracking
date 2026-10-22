# MTMC Monitor — frontend

Interface de monitoramento (React 19 + TypeScript + Vite 8 + Tailwind 4 + shadcn/ui + lucide). Serviço **independente**
(contêiner nginx, sem Python/PyTorch/OpenCV) que consome a API do backend (`mtmc serve`). Não executa YOLO/ByteTrack.

## Arquitetura

```
navegador ──HTTP/WS──► nginx (frontend:8080) ──/api/*──► mtmc-api (FastAPI :8000) ── YOLO+ByteTrack (GPU) ── vídeos
```
* `GET /api/health`, `GET /api/cameras` (polling a cada 2 s; lista **dinâmica**, estado confirmado pelo backend),
  `GET /api/cameras/{id}/snapshot.jpg`, `WS /api/cameras/{id}/ws`.
* **Um WebSocket binário por câmera**; cada mensagem = `[uint32 BE tamanho do header][header JSON][JPEG]`.
  O header traz `frame`, `t_global`, `loop`, tamanho original e as detecções
  (`id` do ByteTrack, `cls`, `name`, `conf`, `bbox` em pixels do vídeo original, `gid`/`uncertain` se houver associação MTMC).
  Sem base64 e sem fila: o servidor só envia o frame **mais recente**.
* **Decisão: desenhar no cliente (canvas)**, não anotar no backend. O backend já tem o frame + detecções estruturadas; reenviar
  frames anotados exigiria nova codificação por modo de exibição, e o canvas permite ligar/desligar caixas, rótulos e trilhas sem
  custo no servidor, com texto nítido. Frame e metadados chegam juntos (mesmo pacote), então não há dessincronização.
  (Observação: o backend não usa `supervision`; o ByteTrack vem do Ultralytics.)
* **Trajetórias**: `TrailStore` por painel guarda a base da caixa (pés) por ID de tracker, máx. 40 pontos por trilha, descarta
  trilhas ausentes por >30 frames e zera quando o vídeo reinicia (novo `loop`). IDs locais de câmeras diferentes nunca se misturam.
  Os IDs mostrados são **IDs locais do ByteTrack**; o ID global MTMC (`G0001`) só aparece se existir associação (modo `replay`).

```
src/
  services/api/        cliente REST (timeout, ApiError)        services/streaming/  CameraStream (WS, reconexão, watchdog) + StreamManager (1 conexão/câmera)
  features/tracking/   trails.ts, drawOverlay.ts               features/monitoring/ página + layouts      features/camera-selection/ seleção
  hooks/               useCameras, useCameraStream, useVisible  components/{layout,camera,status,ui}/  tipos em types/
tests/unit (Vitest) · tests/e2e (Playwright, contra o stack real)
```

## Comportamento relevante
* Remover uma câmera do grid **não** para a inferência; só libera o WebSocket (após 1,5 s de carência).
* Painéis fora da viewport ou aba oculta liberam o fluxo; layout não reabre conexões (contagem de referências).
* Só o último frame é decodificado (`createImageBitmap`); UI de estatísticas atualiza a 2,5 Hz; canvas é imperativo.
* "Ao vivo" exige backend `running` **e** frames chegando ao navegador. "Sem detecções neste frame" é distinto de falha.
* Reconexão exponencial (0,5 s → 8 s) e watchdog de 5 s sem mensagens.
* Layouts 1/2/4/6/9/auto; colunas limitadas pela largura. Layout fixo com menos vagas que câmeras selecionadas mostra aviso.

## Executar
Desenvolvimento (backend em `http://localhost:8000`, ver `.env.example`):
```bash
npm install
npm run dev          # http://localhost:5173 ; /api é encaminhado ao backend (DEV_BACKEND_URL)
npm run typecheck && npm run lint && npm test && npm run build
npm run e2e          # requer o stack rodando; E2E_BASE_URL (padrão :8088), E2E_BROWSER_CHANNEL (padrão msedge)
```
Contêiner (a partir da raiz do repositório):
```bash
docker compose up -d --build            # mtmc-api (GPU) + frontend
docker compose up -d --build frontend   # reconstrói/reinicia só o frontend
docker compose stop frontend
```
| Variável | Serviço | Padrão | Função |
|---|---|---|---|
| `FRONTEND_PORT` | compose | 8080 | porta publicada do frontend |
| `API_PORT` | compose | 8000 | porta publicada (127.0.0.1) da API, uso direto opcional |
| `BACKEND_URL` | frontend | `http://mtmc-api:8000` | destino do proxy `/api` no nginx |
| `DNS_RESOLVER` | frontend | `127.0.0.11` | resolvedor do Docker (o frontend sobe mesmo com o backend parado) |
| `API_BASE_URL` | frontend | `/api` | base usada pelo navegador (gerada em `/config.js` no start) |
| `APP_ENVIRONMENT` | frontend | `local` | rótulo na barra superior |
| `MTMC_CONFIG`, `MTMC_MODE` | mtmc-api | `configs/s01_gpu.yaml`, `live` | config do backend; `live` (YOLO+ByteTrack) ou `replay` (resultados salvos, sem GPU) |

## Testes realizados
* Vitest: 41 testes (layouts, trilhas, desenho de caixas/rótulos/trilhas, protocolo, reconexão e timeout do stream, contagem de
  referências, erros da API, grid com 1–12 câmeras, seleção/busca, layout, ampliar, estados de erro/reconexão, metadados).
* Playwright (Edge, contra o backend real com GPU): 6 testes (descoberta de câmeras, canvas com pixels, seleção, layouts,
  ampliar, sem WebSockets novos ao trocar layout, API fora do ar, HTTP 500).
* Verificação manual no navegador: grid ao vivo, ampliar, queda e retorno do backend (reconecta sozinho).

## Limitações conhecidas
* Backend ao vivo com 5 câmeras simultâneas em uma RTX 2060 (`imgsz` 960): ≈ 5–7 fps processados por câmera e ≈ 50 % dos frames
  descartados para acompanhar o tempo real (inferência 110–180 ms sob contenção; em lote, uma câmera sozinha ≈ 25 fps). Frames
  descartados espaçam os pontos das trilhas e favorecem trocas de ID do ByteTrack. Reduzir `detector.imgsz` aumenta o fps.
* Não foram medidos: consumo de memória do navegador, limites de câmeras além das 5 do dataset, latência ponta a ponta.
* IDs globais ao vivo (`G0052`) vêm do associador do backend (Re-ID + homografia); aparecem só para veículos ligados entre >= 2 câmeras, com atraso de alguns segundos, e podem mudar entre rodadas. Qualidade ao vivo não medida com GT (ver README da raiz).
* Sem autenticação nem TLS; destinado a rede local.
* Todos os vídeos reiniciam juntos em loop (relógio global); não há controle de reprodução/pausa na UI.
* O detector (yolov8n) falha em veículos grandes/próximos em algumas câmeras (visível em c005); limitação do modelo, não da UI.

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

## Interface moderna: recursos e decisões
* **Tema e layout**: preto/grafite/cinza, ciano para ação e seleção, verde/amarelo/vermelho para saúde; vídeo em toda a área do tile; controles sobrepostos só no
  hover/foco (e sempre visíveis em telas de toque). Sem transições sobre o vídeo; `prefers-reduced-motion` respeitado.
* **Drag-and-drop**: `@dnd-kit/core` 6.x (estável; a API nova `@dnd-kit/react` ainda é 0.x). Semântica de **troca** entre duas câmeras; ordem persistida em
  `localStorage` (`mtmc.ui.v1.order`) e mesclada com as câmeras descobertas (novas no fim). O item arrastado é o próprio painel (nenhum clone com canvas).
  Teclado: o `KeyboardSensor` do dnd-kit está ativo na alça (Espaço + setas), mas **não foi validado** em navegador (só o mouse foi testado de ponta a ponta).
* **Painéis ocultos**: tiles fora do layout ou fora da maximização continuam montados (até 16) com a conexão mantida e sem decodificar/desenhar; acima de 16, os
  excedentes são desmontados e reabrem a conexão quando voltam.
* **Persistência**: `mtmc.ui.v1` no `localStorage` com validação/sanitização (dados corrompidos caem nos padrões). Seleção de câmeras continua por sessão.
* **Dependências novas**: `@dnd-kit/core` (arraste e soltura acessível e com sensores configuráveis) e os componentes shadcn `alert-dialog` e `popover`
  (confirmação e ajuda). Removida a dependência órfã `cn` (a CLI do shadcn a adicionou por engano).

## Painel "Modelos e limiares" (somente leitura)
Botão no cabeçalho → painel lateral recolhível (as câmeras continuam visíveis). Abas: Visão geral, Modelos, Limiares e parâmetros,
Por câmera, Histórico. Consulta `GET /api/runtime` apenas com o painel aberto (a cada 5 s), sem tocar nos streams ou na inferência.

**Contrato (`mtmc/runtime_info.py`)** — resposta JSON: `read_only`, `revision` (hash da configuração efetiva), `config`
(`file`, `file_mtime`, `sha256`, `changed_on_disk`, `diff[]`), `versions` (python/torch/tensorrt/onnxruntime/cuda/gpu),
`models[]` (papel, formato **do runtime**, dispositivo, entradas, classes, câmeras que usam), `parameters[]` (`configured`, `applied`,
`status` ∈ confirmed|unconfirmed|diverges, `scope`, `category`, `description`, `fixed_in_code`), `cameras[]` (valores efetivos e origem),
`history.supported=false`.

Regras: **configurado** = arquivo YAML relido do disco; **aplicado** = lido do objeto em execução (`predictor.args` do Ultralytics,
argumentos do `BYTETracker` instanciado, sessão ONNX, `ICudaEngine`). Se o runtime não expõe o valor ⇒ "não confirmado pelo runtime".
Formato/dispositivo vêm de metadados do runtime (`AutoBackend.format`, engine TensorRT, provedores do onnxruntime), nunca do nome do
arquivo. Valores fixos no código (ex.: NMS 0,7 no caminho unificado) aparecem marcados "fixo no código". A precisão FP16 do engine
é um metadado gravado pelo nosso exportador e é exibida como tal (não confirmada pelo runtime). Se o arquivo de configuração mudou
depois da partida, a Visão geral mostra o par rodando × disco (só vale após reiniciar o backend).

Não há parâmetros por câmera sobrescritos hoje (`detector`/`tracker` são globais); por câmera existem `offset_adjust`, `drift_ppm`,
zonas e homografia, mostrados com a origem (herdado/sobrescrito/metadados). Histórico de alterações: **não existe** no backend.

### Etapa 2 (proposta, NÃO implementada): edição
Exigiria um contrato novo e seguro no backend antes de qualquer botão "Salvar":
1. `PATCH /api/runtime/parameters` com `{id, value, expected_revision}`; o backend valida tipo/mín/máx por parâmetro (esquema declarado
   no próprio backend) e rejeita se `expected_revision` não for a atual (evita sobrescrever edição concorrente).
2. Cada parâmetro declara `apply: "hot" | "reload_model" | "restart_component"`; a UI mostra isso e pede confirmação com valores
   anterior e novo antes de aplicar.
3. Persistência explícita (arquivo de overrides versionado, não editar o YAML base) e trilha de auditoria
   (`GET /api/runtime/history`: usuário, valor anterior/novo, data, revisão).
4. A UI só declara "aplicado" depois que o `GET /api/runtime` seguinte confirmar o valor no objeto em execução.
Autenticação/autorização é pré-requisito (o serviço hoje não tem).

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
* Vitest: 70 testes (41 do monitor, 9 do painel de modelos e limiares, 20 da interface: ordem, grid, preferências, atalhos, divisores, sidebar, overlays) (layouts, trilhas, desenho de caixas/rótulos/trilhas, protocolo, reconexão e timeout do stream, contagem de
  referências, erros da API, grid com 1–12 câmeras, seleção/busca, layout, ampliar, estados de erro/reconexão, metadados).
* Playwright (Edge, contra o backend real com GPU): 21 testes (6 do monitor, 3 do painel e 12 da interface: arrastar com mouse real, sem remontar/reabrir streams, persistência após recarregar, barras, atalhos, console sem erros, 4 resoluções incluindo 3440×1440 e 390×844) (descoberta de câmeras, canvas com pixels, seleção, layouts,
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

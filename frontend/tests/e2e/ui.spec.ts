import { expect, test, type Page } from '@playwright/test'

const live = (page: Page) => page.locator('[data-testid^="panel-"] [data-health="live"]')
const tileIds = (page: Page) =>
  page.locator('[data-testid^="tile-"]:visible').evaluateAll((els) => els.map((e) => (e.getAttribute('data-testid') ?? '').replace('tile-', '')))
const instances = (page: Page) =>
  page.locator('[data-testid^="panel-"]').evaluateAll((els) => els.map((e) => `${e.getAttribute('data-testid')}=${e.getAttribute('data-instance')}`).sort())

async function ready(page: Page) {
  await page.goto('/')
  await expect(live(page).first()).toBeVisible({ timeout: 45_000 })
  await expect(page.locator('[data-testid^="tile-"]:visible').first()).toBeVisible()
}

/** Arrasta com o mouse REAL (down → movimentos graduais → up), do centro do alvo de origem ao centro do alvo de destino. */
async function dragTo(page: Page, from: string, to: string) {
  const h = await page.locator(`[data-testid="tile-${from}"] [data-testid="drag-handle"]`).boundingBox()
  const t = await page.locator(`[data-testid="tile-${to}"]`).boundingBox()
  if (!h || !t) throw new Error('alça ou destino não encontrados')
  await page.mouse.move(h.x + h.width / 2, h.y + h.height / 2)
  await page.mouse.down()
  await page.mouse.move(h.x + h.width / 2 + 12, h.y + h.height / 2 + 12, { steps: 4 }) // passa do limiar de ativação (6 px)
  await page.mouse.move(t.x + t.width / 2, t.y + t.height / 2, { steps: 12 })
  return async () => page.mouse.up()
}

test.describe('interface modernizada', () => {
  test('carrega sem erros de console, exceções nem requisições com falha', async ({ page }) => {
    const errors: string[] = []
    page.on('console', (m) => m.type() === 'error' && errors.push(`console: ${m.text()}`))
    page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`))
    page.on('requestfailed', (r) => errors.push(`requestfailed: ${r.method()} ${r.url()} ${r.failure()?.errorText}`))
    page.on('response', (r) => r.status() >= 400 && errors.push(`http ${r.status()}: ${r.url()}`))
    await ready(page)
    await page.getByTestId('open-processing').click()
    await expect(page.getByTestId('tab-overview')).toBeVisible()
    await page.waitForTimeout(3000)
    expect(errors, errors.join('\n')).toEqual([])
  })

  test('drag-and-drop com mouse troca duas câmeras sem remontar nem reabrir streams, e persiste', async ({ page }) => {
    const sockets: string[] = []
    page.on('websocket', (ws) => sockets.push(ws.url()))
    await ready(page)
    const ids = await tileIds(page)
    expect(ids.length).toBeGreaterThanOrEqual(3)
    const [a, b] = ids
    const instBefore = await instances(page)
    const wsBefore = sockets.length

    const release = await dragTo(page, a, b)
    // durante o arraste: indicador de destino visível e câmera arrastada destacada
    await expect(page.locator(`[data-testid="tile-${b}"][data-drop-target="true"]`)).toBeVisible()
    await expect(page.locator(`[data-testid="tile-${a}"][data-dragging="true"]`)).toBeVisible()
    await expect(page.getByText(`Trocar com ${b}`)).toBeVisible()
    await release()

    const after = await tileIds(page)
    expect(after.slice(0, 2)).toEqual([b, a]) // trocaram
    expect(after.slice(2)).toEqual(ids.slice(2)) // as demais ficaram onde estavam
    expect(await instances(page)).toEqual(instBefore) // nenhum painel foi remontado
    expect(sockets.length).toBe(wsBefore) // nenhuma conexão nova
    await expect(live(page).first()).toBeVisible()

    await page.reload()
    await expect(page.locator('[data-testid^="tile-"]:visible').first()).toBeVisible()
    expect((await tileIds(page)).slice(0, 2)).toEqual([b, a]) // ordem restaurada
  })

  test('clicar em botões não inicia arraste; a ordem permanece estável ao desmarcar e selecionar', async ({ page }) => {
    await ready(page)
    const ids = await tileIds(page)
    const first = ids[0]
    await page.locator(`[data-testid="tile-${first}"]`).hover()
    await page.getByTestId(`expand-${first}`).click() // clique no botão: não arrasta
    await expect(page.getByTestId('video-grid')).toHaveAttribute('data-count', '1')
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('video-grid')).toHaveAttribute('data-count', String(ids.length))
    expect(await tileIds(page)).toEqual(ids)

    await page.getByRole('checkbox', { name: `Selecionar ${ids[1]}` }).click()
    expect(await tileIds(page)).toEqual(ids.filter((x) => x !== ids[1]))
    await page.getByRole('checkbox', { name: `Selecionar ${ids[1]}` }).click()
    expect(await tileIds(page)).toEqual(ids) // volta exatamente à posição de antes
  })

  test('maximizar/restaurar e trocar de layout não reiniciam streams', async ({ page }) => {
    const sockets: string[] = []
    page.on('websocket', (ws) => sockets.push(ws.url()))
    await ready(page)
    const inst = await instances(page)
    const ws0 = sockets.length
    const id = (await tileIds(page))[1]
    await page.locator(`[data-testid="tile-${id}"]`).hover()
    await page.getByTestId(`expand-${id}`).click()
    await expect(page.getByTestId('video-grid')).toHaveAttribute('data-count', '1')
    await page.keyboard.press('Escape')
    for (const k of ['2', '4', '1', 'a']) {
      await page.keyboard.press(k)
      await page.waitForTimeout(150)
    }
    expect(await instances(page)).toEqual(inst)
    expect(sockets.length).toBe(ws0)
    await expect(live(page).first()).toBeVisible()
  })

  test('lista de câmeras: recolhe (modo compacto), redimensiona com o mouse e persiste', async ({ page }) => {
    await ready(page)
    const sb = page.getByTestId('sidebar')
    const w0 = (await sb.boundingBox())!.width
    // redimensionar arrastando o divisor
    const sp = (await page.getByTestId('splitter-grow-right').boundingBox())!
    await page.mouse.move(sp.x + sp.width / 2, sp.y + 200)
    await page.mouse.down()
    await page.mouse.move(sp.x + sp.width / 2 + 70, sp.y + 200, { steps: 6 })
    await page.mouse.up()
    const w1 = (await sb.boundingBox())!.width
    expect(w1).toBeGreaterThan(w0 + 40)
    // recolher libera espaço para o grid
    const gridW0 = (await page.getByTestId('video-grid').boundingBox())!.width
    await page.getByTestId('sidebar-toggle').click()
    await expect(sb).toHaveAttribute('data-collapsed', 'true')
    expect((await sb.boundingBox())!.width).toBeLessThan(70)
    await expect.poll(async () => (await page.getByTestId('video-grid').boundingBox())!.width).toBeGreaterThan(gridW0 + 100)
    await expect(live(page).first()).toBeVisible() // seleção e processamento preservados
    // persistência
    await page.reload()
    await expect(page.getByTestId('sidebar')).toHaveAttribute('data-collapsed', 'true')
    await page.keyboard.press('b')
    await expect(page.getByTestId('sidebar')).toHaveAttribute('data-collapsed', 'false')
    expect(Math.abs((await page.getByTestId('sidebar').boundingBox())!.width - w1)).toBeLessThan(3) // largura restaurada
  })

  test('painel de configurações: abre/fecha (P), redimensiona, preserva a aba e continua mostrando valores reais', async ({ page }) => {
    await ready(page)
    await page.keyboard.press('p')
    const panel = page.getByTestId('processing-panel')
    await expect(panel).toHaveAttribute('data-open', 'true')
    await panel.getByRole('tab', { name: 'Limiares e parâmetros' }).click()
    await expect(panel.getByTestId('param-detector.conf')).toContainText('confirmado')
    const w0 = (await panel.boundingBox())!.width
    const sp = (await page.getByTestId('splitter-grow-left').boundingBox())!
    await page.mouse.move(sp.x + sp.width / 2, sp.y + 200)
    await page.mouse.down()
    await page.mouse.move(sp.x + sp.width / 2 - 90, sp.y + 200, { steps: 6 })
    await page.mouse.up()
    expect((await panel.boundingBox())!.width).toBeGreaterThan(w0 + 50)
    await page.keyboard.press('p')
    await expect(panel).toHaveAttribute('data-open', 'false')
    await expect(page.getByTestId('edge-tab-processing')).toBeVisible() // controle visível para recuperar
    await page.getByTestId('edge-tab-processing').click()
    await expect(panel.getByTestId('param-detector.conf')).toBeVisible() // mesma aba, conteúdo preservado
    await page.reload()
    await expect(panel).toHaveAttribute('data-open', 'true')
    await expect(panel.getByRole('tab', { name: 'Limiares e parâmetros' })).toHaveAttribute('data-state', 'active')
  })

  test('sobreposições por câmera (hover) e globais funcionam e persistem', async ({ page }) => {
    await ready(page)
    const ids = await tileIds(page)
    const a = ids[0]
    const b = ids[1]
    await page.locator(`[data-testid="tile-${a}"]`).hover()
    const t = page.getByTestId(`tgl-trails-${a}`)
    await expect(t).toHaveAttribute('aria-pressed', 'true')
    await t.click()
    await expect(t).toHaveAttribute('aria-pressed', 'false')
    await expect(page.getByTestId(`tgl-trails-${b}`)).toHaveAttribute('aria-pressed', 'true')
    await page.reload()
    await expect(page.locator('[data-testid^="tile-"]:visible').first()).toBeVisible()
    await expect(page.getByTestId(`tgl-trails-${a}`)).toHaveAttribute('aria-pressed', 'false')
    await page.getByTestId('global-boxes').click() // "todas as câmeras"
    await expect(page.getByTestId(`tgl-boxes-${b}`)).toHaveAttribute('aria-pressed', 'false')
  })

  test('barra superior mostra conectadas, falhas e desempenho reais', async ({ page }) => {
    await ready(page)
    const total = Number(await page.getByTestId('count-total').innerText())
    await expect(page.getByTestId('count-processing')).toHaveText(String(total))
    await expect(page.getByTestId('count-failed')).toHaveText('0')
    await expect(page.getByTestId('metric-perf')).toContainText(/\d+\.\d\s*fps/)
  })

  for (const [w, h] of [[1920, 1080], [3440, 1440], [1366, 768], [390, 844]] as const) {
    test(`funciona em ${w}×${h}: sem rolagem horizontal e com vídeo ao vivo`, async ({ page }) => {
      await page.setViewportSize({ width: w, height: h })
      await ready(page)
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
      expect(overflow).toBeLessThanOrEqual(1)
      const box = (await page.locator('[data-testid^="tile-"]:visible').first().boundingBox())!
      expect(box.width).toBeGreaterThan(120)
      expect(box.height).toBeGreaterThan(70)
      // o grid usa o espaço: nenhuma linha/coluna sobra vazia além de uma célula
      const { cols, count } = await page.getByTestId('video-grid').evaluate((e) => ({ cols: Number(e.getAttribute('data-cols')), count: Number(e.getAttribute('data-count')) }))
      expect(cols).toBeGreaterThan(0)
      expect(Math.ceil(count / cols) * cols - count).toBeLessThan(cols)
      if (w >= 3000) expect(cols).toBeGreaterThanOrEqual(3) // ultrawide: mais colunas
      if (w < 600) await expect(page.getByTestId('sidebar')).toHaveAttribute('data-collapsed', 'true')
    })
  }
})

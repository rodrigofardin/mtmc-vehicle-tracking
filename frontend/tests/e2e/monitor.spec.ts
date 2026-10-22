import { expect, test, type Page } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => sessionStorage.clear())
})

// painéis EXIBIDOS (tiles ocultos por layout/maximização continuam montados, mas não visíveis)
const panels = (page: Page) => page.locator('[data-testid^="panel-"]:visible')

const live = (page: Page) => page.locator('[data-testid^="panel-"] [data-health="live"]')

/** n = quantidade exata de painéis "ao vivo"; sem n, basta ao menos um. */
async function waitLive(page: Page, n?: number) {
  if (n === undefined) await expect(live(page).first()).toBeVisible({ timeout: 45_000 })
  else await expect(live(page)).toHaveCount(n, { timeout: 45_000 })
}

test('descobre as câmeras reais e exibe vídeo com detecções', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('backend-status')).toHaveAttribute('data-connection', 'online')
  const total = Number(await page.getByTestId('count-total').innerText())
  expect(total).toBeGreaterThan(0)
  await expect(panels(page)).toHaveCount(total)
  await waitLive(page, total)

  // o canvas realmente recebeu pixels (frame JPEG decodificado e desenhado)
  const painted = await page.locator('canvas').first().evaluate((c: HTMLCanvasElement) => {
    const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data
    let sum = 0
    for (let i = 0; i < d.length; i += 4 * 97) sum += d[i] + d[i + 1] + d[i + 2]
    return c.width > 0 && sum > 0
  })
  expect(painted).toBe(true)
  await expect(page.locator('[data-testid^="stats-"]').first()).toContainText(/obj/)
})

test('seleção e layout adaptam o grid', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('count-total')).not.toHaveText('0')
  const total = Number(await page.getByTestId('count-total').innerText())

  await page.getByRole('button', { name: 'Limpar' }).click()
  await page.getByTestId('confirm-clear').click()
  await expect(page.getByText('Nenhuma câmera selecionada')).toBeVisible()
  await expect(panels(page)).toHaveCount(0)

  await page.getByRole('button', { name: 'Selecionar todas' }).click()
  await expect(panels(page)).toHaveCount(total)

  const first = (await page.locator('[data-testid^="cam-item-"]').first().getAttribute('data-testid'))!.replace('cam-item-', '')
  await page.getByRole('checkbox', { name: `Selecionar ${first}` }).click()
  await expect(panels(page)).toHaveCount(total - 1)

  await page.getByTestId('layout-1').click()
  await expect(panels(page)).toHaveCount(1)
  await page.getByTestId('layout-auto').click()
  await expect(panels(page)).toHaveCount(total - 1)
})

test('ampliar uma câmera e voltar', async ({ page }) => {
  await page.goto('/')
  await waitLive(page)
  const n = await panels(page).count()
  await page.getByRole('button', { name: /^Maximizar / }).first().click()
  await expect(panels(page)).toHaveCount(1)
  await page.keyboard.press('Escape')
  await expect(panels(page)).toHaveCount(n)
})

test('mudar de layout não abre conexões WebSocket novas', async ({ page }) => {
  const sockets: string[] = []
  page.on('websocket', (ws) => sockets.push(ws.url()))
  await page.goto('/')
  await waitLive(page)
  const before = sockets.length
  await page.getByTestId('layout-2').click()
  await page.getByTestId('layout-auto').click()
  await page.waitForTimeout(1500)
  expect(sockets.length).toBe(before)
})

test('API fora do ar: mensagem clara, sem travar', async ({ page }) => {
  await page.route('**/api/**', (r) => r.abort())
  await page.goto('/')
  await expect(page.getByRole('main').getByText('Backend indisponível')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Tentar novamente' })).toBeVisible()
})

test('erro HTTP da API é tratado', async ({ page }) => {
  await page.route('**/api/cameras', (r) => r.fulfill({ status: 500, body: 'boom' }))
  await page.goto('/')
  await expect(page.getByTestId('backend-status')).toHaveAttribute('data-connection', 'offline')
})

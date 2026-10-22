import { expect, test, type Page } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => sessionStorage.clear())
})

const live = (page: Page) => page.locator('[data-testid^="panel-"] [data-health="live"]')

test('abre o painel com dados reais do backend, sem derrubar as câmeras', async ({ page }) => {
  await page.goto('/')
  await expect(live(page).first()).toBeVisible({ timeout: 45_000 })
  const before = await page.locator('[data-testid^="panel-"]:visible').count()

  await page.getByTestId('open-processing').click()
  const panel = page.getByTestId('processing-panel')
  await expect(panel).toBeVisible()
  await expect(panel.getByTestId('tab-overview')).toContainText('Revisão')
  await expect(panel.getByText('somente leitura').first()).toBeVisible()
  // câmeras continuam visíveis e ao vivo ao lado do painel
  await expect(page.locator('[data-testid^="panel-"]:visible')).toHaveCount(before)
  await expect(live(page).first()).toBeVisible()

  // Modelos: backend real em modo ao vivo expõe o detector com formato vindo do runtime
  await panel.getByRole('tab', { name: 'Modelos' }).click()
  const det = panel.getByTestId('model-detector')
  await expect(det).toBeVisible()
  await det.getByRole('button').click()
  await expect(det).toContainText('Formato (do runtime)')
  await expect(det).toContainText(/TensorRT|PyTorch|ONNX/)
  await expect(panel.getByTestId('class-list')).toContainText('car')

  // Parâmetros: confiança mínima confirmada pelo runtime; filtro de estado funciona
  await panel.getByRole('tab', { name: 'Limiares e parâmetros' }).click()
  await expect(panel.getByTestId('param-detector.conf')).toContainText('confirmado')
  const total = Number((await panel.getByTestId('param-count').innerText()).split(' de ')[1])
  expect(total).toBeGreaterThan(20)
  await panel.getByLabel('estado').selectOption('unconfirmed')
  await expect(panel.getByTestId('param-count')).not.toContainText(`${total} de`)
  await expect(panel.getByText('não confirmado pelo runtime').first()).toBeVisible()

  // Por câmera: uma coluna por câmera descoberta
  await panel.getByRole('tab', { name: 'Por câmera' }).click()
  await expect(panel.getByTestId('tab-cameras').getByRole('columnheader', { name: /^c\d{3}$/ })).toHaveCount(before)

  // Histórico: sem suporte no backend
  await panel.getByRole('tab', { name: 'Histórico' }).click()
  await expect(panel.getByTestId('tab-history')).toContainText('Sem histórico')
})

test('consultar parâmetros não interrompe o processamento nem os streams', async ({ page, request }) => {
  await page.goto('/')
  await expect(live(page).first()).toBeVisible({ timeout: 45_000 })
  const frames = async () => (await (await request.get('/api/cameras')).json()).map((c: { frames_published: number }) => c.frames_published)
  const a = await frames()
  await page.getByTestId('open-processing').click()
  await expect(page.getByTestId('tab-overview')).toBeVisible()
  for (let i = 0; i < 10; i++) await request.get('/api/runtime') // rajada de consultas
  await page.waitForTimeout(3000)
  const b = await frames()
  expect(b.every((n: number, i: number) => n > a[i])).toBe(true) // todas as câmeras seguiram publicando frames
  await expect(live(page).first()).toBeVisible()
})

test('erro do endpoint /runtime é exibido sem quebrar a página', async ({ page }) => {
  await page.route('**/api/runtime', (r) => r.fulfill({ status: 500, body: 'boom' }))
  await page.goto('/')
  await page.getByTestId('open-processing').click()
  await expect(page.getByTestId('processing-panel').getByRole('alert')).toContainText('Não foi possível consultar')
  await expect(page.locator('[data-testid^="panel-"]').first()).toBeVisible()
})

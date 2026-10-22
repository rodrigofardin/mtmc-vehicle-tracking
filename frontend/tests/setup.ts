import '@testing-library/jest-dom/vitest'

// jsdom não implementa ResizeObserver (usado pelo Radix ScrollArea e pelo grid)
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver

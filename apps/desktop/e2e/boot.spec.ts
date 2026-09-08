/**
 * E2E smoke tests for the dev-mode desktop app.
 *
 * These tests launch the Electron app from the built dist/ (not the
 * packaged binary) with a real `hermes serve` backend pointed at a mock
 * inference server. The full chain is exercised:
 *
 *   electron → hermes serve (python) → mock provider → renderer
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 * Run from the nix devshell:
 *   npm exec playwright test e2e/boot.spec.ts --reporter=list
 */
import { expect, test } from './test'

import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'
import { expectVisualSnapshot } from './visual-snapshot'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await fixture.app.evaluate(async ({ app, BrowserWindow }, logPath) => {
    const fs = process.getBuiltinModule('node:fs')
    const record = (event: string, detail: unknown = null) => {
      const handles = (process as unknown as { _getActiveHandles(): { constructor: {name: string}; pid?: number; exitCode?: number }[] })._getActiveHandles()
      fs.appendFileSync(logPath, JSON.stringify({time: Date.now(), pid: process.pid, event, detail,
        windows: BrowserWindow.getAllWindows().map(w => ({id:w.id,destroyed:w.isDestroyed()})),
        handles: handles.map(h=>({type:h.constructor.name,pid:h.pid,exitCode:h.exitCode}))})+'\n')
    }
    record('installed')
    for (const event of ['before-quit','will-quit','quit','window-all-closed'] as const) {
      app.on(event, (e: { defaultPrevented?: boolean }) => record(event, {prevented:e?.defaultPrevented}))
    }
    for (const win of BrowserWindow.getAllWindows()) {
      for (const event of ['close','closed','unresponsive','responsive'] as const) {
        win.on(event, (e: {defaultPrevented?: boolean}) => record('window.'+event,{prevented:e?.defaultPrevented}))
      }
      win.webContents.on('will-prevent-unload', ()=>record('will-prevent-unload'))
      win.webContents.on('render-process-gone', (_e,d)=>record('render-process-gone',d))
    }
  }, process.env.SHUTDOWN_DIAG_LOG!)

})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test.describe('dev-mode boot with mock backend', () => {
  test('window opens with Hermes title', async () => {
    const title = await fixture!.page.title()
    expect(title).toContain('Hermes')
  })

  test('renderer mounts and shows DOM content', async () => {
    const page = fixture!.page
    // Wait for the React root to mount. The app renders into #root
    // (see src/main.tsx), but content may arrive through portals — so
    // check the body for any interactive content instead.
    await page.waitForSelector('body', { state: 'attached' })
    // Wait for the main app shell — the composer is always present.
    await page.waitForSelector('textarea, [contenteditable="true"]', {
      state: 'attached',
      timeout: 30_000,
    })
  })

  // A preload that throws never reaches contextBridge, so the renderer boots
  // into "Desktop IPC bridge is unavailable" and every test below it dies on a
  // 120s never-became-ready timeout instead. Checking the bridge by name makes
  // that failure legible. The sandbox lets preload require only electron,
  // events, timers and url — adding any other node builtin lands here.
  test('the preload bridge reaches the renderer', async () => {
    const bridge = await fixture!.page.evaluate(() => {
      const desktop = (window as unknown as { hermesDesktop?: Record<string, unknown> }).hermesDesktop

      return {
        present: typeof desktop,
        glassSupported: typeof desktop?.glassSupported,
        translucencySupported: typeof desktop?.translucencySupported
      }
    })

    expect(bridge).toEqual({ present: 'object', glassSupported: 'boolean', translucencySupported: 'boolean' })
  })

  test('backend boots and app becomes ready', async () => {
    // This is the big one — wait for the full boot chain to complete:
    // electron starts → hermes serve is spawned → WS connects → config
    // loaded → sessions loaded → boot overlay dismissed → composer visible.
    await waitForAppReady(fixture!, 120_000)
  })

  test('screenshot after boot', async () => {
    await expectVisualSnapshot(fixture!.page, { name: 'boot-ready', app: fixture!.app })
  })
})

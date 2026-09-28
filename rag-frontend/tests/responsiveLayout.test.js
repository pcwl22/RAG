import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'

const appSource = await readFile(new URL('../src/App.vue', import.meta.url), 'utf8')
const sidebarSource = await readFile(
  new URL('../src/components/ChatSidebar.vue', import.meta.url),
  'utf8'
)

test('chat workspace can shrink without clipping its controls', () => {
  assert.match(appSource, /\.chat-area\s*\{[^}]*min-width:\s*0;/s)
  assert.match(appSource, /textarea\s*\{[^}]*min-width:\s*0;/s)
  assert.match(appSource, /@media \(max-width: 768px\)[\s\S]*?\.main-container\s*\{[^}]*flex-direction:\s*column;/)
})

test('mobile sidebar becomes a bounded horizontal session strip', () => {
  assert.match(sidebarSource, /@media \(max-width: 768px\)/)
  assert.match(sidebarSource, /\.sidebar\s*\{[^}]*width:\s*100%;[^}]*max-height:/s)
  assert.match(sidebarSource, /\.chat-list\s*\{[^}]*overflow-x:\s*auto;/s)
  assert.match(sidebarSource, /\.chat-item\s*\{[^}]*flex:\s*0 0 220px;/s)
})

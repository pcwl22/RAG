import test from 'node:test'
import assert from 'node:assert/strict'

import { getModelContextText, getParentContextText, getRelevantSnippet } from '../src/utils/contextText.js'
import { formatDuration, formatScore, truncateText } from '../src/utils/formatters.js'
import { fetchHealth } from '../src/api/ragApi.js'
import { readEventStream } from '../src/utils/eventStream.js'

test('context helpers preserve model-visible child content', () => {
  const context = {
    content: 'stored child',
    child_content: 'model child',
    metadata: { parent_content: 'parent text' }
  }
  assert.equal(getModelContextText(context), 'model child')
  assert.equal(getParentContextText(context), 'parent text')
  assert.match(getRelevantSnippet('前文劳动合同解除后文', '劳动合同'), /劳动合同/)
})

test('formatters handle boundaries', () => {
  assert.equal(truncateText('abcdef', 3), 'abc...')
  assert.equal(formatDuration(0.25), '250ms')
  assert.equal(formatScore(0.12345), '0.1235')
})

test('health requires the readiness endpoint to report ready', async () => {
  const originalFetch = globalThis.fetch
  try {
    let requestedUrl = ''
    globalThis.fetch = async url => {
      requestedUrl = String(url)
      return new Response(JSON.stringify({ status: 'ready' }), {
        status: 200,
        headers: { 'content-type': 'application/json' }
      })
    }
    assert.equal(await fetchHealth(), true)
    assert.match(requestedUrl, /\/health\/ready$/)

    globalThis.fetch = async () => new Response(
      JSON.stringify({ status: 'not_ready' }),
      { status: 503, headers: { 'content-type': 'application/json' } }
    )
    assert.equal(await fetchHealth(), false)
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('event stream rejects a connection that closes without done', async () => {
  const response = new Response('data: {"type":"chunk","data":"partial"}\n\n')
  await assert.rejects(
    readEventStream(response, {}),
    /流式响应意外中断/
  )
})

import test from 'node:test'
import assert from 'node:assert/strict'

import { getModelContextText, getParentContextText, getRelevantSnippet } from '../src/utils/contextText.js'
import { formatDuration, formatScore, parseTimestamp, truncateText } from '../src/utils/formatters.js'
import { ApiError, fetchDocumentChunks, fetchDocuments, fetchHealth } from '../src/api/ragApi.js'
import { parseSigninCallback } from '../src/auth/oidc.js'
import { readEventStream } from '../src/utils/eventStream.js'
import { loadSessions, saveSessions } from '../src/utils/chatStorage.js'
import { pollDocumentTask } from '../src/utils/uploadPolling.js'

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

test('timezone-less database timestamps are interpreted as UTC', () => {
  assert.equal(
    parseTimestamp('2026-09-23T09:12:59.125')?.toISOString(),
    '2026-09-23T09:12:59.125Z'
  )
  assert.equal(
    parseTimestamp('2026-09-23T09:12:59.125+08:00')?.toISOString(),
    '2026-09-23T01:12:59.125Z'
  )
  assert.equal(parseTimestamp('not-a-date'), null)
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

test('event stream cancels an interrupted reader before releasing it', async () => {
  let cancelled = false
  let released = false
  let reads = 0
  const encoder = new TextEncoder()
  const reader = {
    async read() {
      reads += 1
      if (reads === 1) {
        return { value: encoder.encode('data: {"type":"chunk","data":"partial"}\n\n'), done: false }
      }
      return { value: undefined, done: true }
    },
    async cancel() {
      cancelled = true
    },
    releaseLock() {
      released = true
    }
  }
  const response = { body: { getReader: () => reader } }

  await assert.rejects(readEventStream(response, {}), /流式响应意外中断/)
  assert.equal(cancelled, true)
  assert.equal(released, true)
})

test('document APIs send bounded pagination parameters', async () => {
  const originalFetch = globalThis.fetch
  const urls = []
  try {
    globalThis.fetch = async url => {
      urls.push(String(url))
      return new Response(JSON.stringify({ total: 0, documents: [], chunks: [] }), {
        status: 200,
        headers: { 'content-type': 'application/json' }
      })
    }
    await fetchDocuments({ skip: 100, limit: 50 })
    await fetchDocumentChunks('doc/one', { skip: 200, limit: 200 })

    assert.match(urls[0], /documents\?skip=100&limit=50$/)
    assert.match(urls[1], /documents\/doc%2Fone\/chunks\?skip=200&limit=200$/)
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('OIDC error callback is recognized and cannot become a redirect loop', () => {
  assert.deepEqual(
    parseSigninCallback('?error=access_denied&error_description=cancelled&state=abc'),
    { kind: 'error', error: 'access_denied', description: 'cancelled' }
  )
  assert.deepEqual(parseSigninCallback('?code=abc&state=def'), { kind: 'success' })
  assert.equal(parseSigninCallback('?error=access_denied'), null)
})

test('chat storage rejects malformed schemas and survives storage failures', () => {
  const malformed = {
    getItem: () => JSON.stringify([null, { id: 'bad', messages: 'not-an-array' }])
  }
  assert.deepEqual(loadSessions(malformed, 'sessions'), [])

  const inaccessible = {
    getItem: () => { throw new Error('blocked') },
    setItem: () => { throw new Error('quota') }
  }
  assert.deepEqual(loadSessions(inaccessible, 'sessions'), [])
  assert.equal(saveSessions(inaccessible, 'sessions', [{ id: 'one', messages: [] }]), false)
})

test('task polling backs off across transient 503 responses and keeps progress', async () => {
  const delays = []
  const progress = []
  let calls = 0
  const result = await pollDocumentTask('task-1', {
    fetchStatus: async () => {
      calls += 1
      if (calls <= 2) throw new ApiError('unavailable', { status: 503 })
      return { status: 'completed', progress: 100 }
    },
    delay: async milliseconds => { delays.push(milliseconds) },
    onProgress: status => progress.push(status.progress),
    baseDelayMs: 100,
    maxPolls: 5
  })

  assert.equal(result.status, 'completed')
  assert.deepEqual(delays, [200, 400])
  assert.deepEqual(progress, [100])
})

test('terminal task failure is not retried', async () => {
  let calls = 0
  await assert.rejects(
    pollDocumentTask('task-2', {
      fetchStatus: async () => {
        calls += 1
        return { status: 'failed', error: 'ingestion failed' }
      },
      delay: async () => {},
      maxPolls: 5
    }),
    error => error.terminal === true
  )
  assert.equal(calls, 1)
})

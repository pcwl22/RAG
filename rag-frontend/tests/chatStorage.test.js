import test from 'node:test'
import assert from 'node:assert/strict'

import { buildChatStorageKey, compactSessions, saveSessions } from '../src/utils/chatStorage.js'

test('chat persistence is scoped by tenant and authenticated subject', () => {
  const alice = buildChatStorageKey({ profile: { tenant_id: 'tenant-a', sub: 'alice' } })
  const bob = buildChatStorageKey({ profile: { tenant_id: 'tenant-a', sub: 'bob' } })
  const otherTenant = buildChatStorageKey({ profile: { tenant_id: 'tenant-b', sub: 'alice' } })

  assert.equal(alice, 'rag_chat_sessions:tenant-a:alice')
  assert.notEqual(alice, bob)
  assert.notEqual(alice, otherTenant)
  assert.equal(buildChatStorageKey(), 'rag_chat_sessions:local')
})

test('chat persistence drops heavy traces and bounds history', () => {
  const messages = Array.from({ length: 105 }, (_, index) => ({
    role: 'assistant', content: `message-${index}`, process: { contexts: ['large'] }
  }))
  const compacted = compactSessions([{ id: '1', title: 'test', messages }])

  assert.equal(compacted[0].messages.length, 100)
  assert.equal(compacted[0].messages[0].content, 'message-5')
  assert.equal('process' in compacted[0].messages[0], false)
})

test('chat persistence handles storage quota failures', () => {
  const storage = { setItem() { throw new Error('quota') } }
  assert.equal(saveSessions(storage, 'key', []), false)
})

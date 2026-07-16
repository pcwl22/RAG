import assert from 'node:assert/strict'
import test from 'node:test'

import { useChatSessions } from '../src/composables/useChatSessions.js'

const createStorage = () => {
  const values = new Map()
  return {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value)
  }
}

test('a completed request updates its originating session after a switch', () => {
  globalThis.localStorage = createStorage()
  const chats = useChatSessions()
  chats.createNewChat()

  const originId = chats.currentSessionId.value
  const originMessages = chats.messages.value
  originMessages.push({ role: 'user', content: '原会话问题' })

  chats.createNewChat()
  const currentId = chats.currentSessionId.value
  originMessages.push({ role: 'assistant', content: '原会话答案' })
  chats.updateCurrentSession(originId, originMessages)

  assert.notEqual(currentId, originId)
  assert.equal(chats.currentSessionId.value, currentId)
  const origin = chats.chatSessions.value.find(session => session.id === originId)
  assert.deepEqual(origin.messages.map(message => message.content), ['原会话问题', '原会话答案'])
})

export const MAX_STORED_SESSIONS = 20
export const MAX_STORED_MESSAGES = 100
const MAX_CONTENT_LENGTH = 20000
const STORAGE_PREFIX = 'rag_chat_sessions'

const storageSegment = value => encodeURIComponent(String(value || 'unknown'))

export const buildChatStorageKey = (user = null) => {
  const profile = user?.profile || {}
  if (!profile.sub) return `${STORAGE_PREFIX}:local`
  return `${STORAGE_PREFIX}:${storageSegment(profile.tenant_id)}:${storageSegment(profile.sub)}`
}

const compactMessage = (message = {}) => ({
  role: message.role === 'assistant' ? 'assistant' : 'user',
  content: String(message.content || '').slice(0, MAX_CONTENT_LENGTH),
  ...(message.createdAt ? { createdAt: message.createdAt } : {})
})

export const compactSessions = (sessions = []) =>
  sessions.slice(0, MAX_STORED_SESSIONS).map(session => ({
    id: session.id,
    title: session.title,
    messageCount: Number(session.messageCount || session.messages?.length || 0),
    createdAt: session.createdAt,
    updatedAt: session.updatedAt,
    messages: (session.messages || []).slice(-MAX_STORED_MESSAGES).map(compactMessage)
  }))

export const saveSessions = (storage, key, sessions) => {
  const compacted = compactSessions(sessions)
  try {
    storage.setItem(key, JSON.stringify(compacted))
    return true
  } catch (error) {
    // Quota may already be mostly consumed by the origin. Retry with only the
    // newest five sessions before giving up without breaking the chat UI.
    try {
      storage.setItem(key, JSON.stringify(compacted.slice(0, 5)))
      return true
    } catch {
      console.warn('Unable to persist chat sessions:', error)
      return false
    }
  }
}

const normalizeMessage = (message) => {
  if (!message || typeof message !== 'object') return null
  if (!['user', 'assistant'].includes(message.role) || typeof message.content !== 'string') {
    return null
  }
  return compactMessage(message)
}

const normalizeSession = (session) => {
  if (
    !session ||
    typeof session !== 'object' ||
    typeof session.id !== 'string' ||
    !session.id ||
    session.id.length > 128
  ) {
    return null
  }
  if (!Array.isArray(session.messages)) return null
  const messages = session.messages
    .slice(-MAX_STORED_MESSAGES)
    .map(normalizeMessage)
    .filter(Boolean)
  return {
    id: session.id,
    title: typeof session.title === 'string' && session.title
      ? session.title.slice(0, 200)
      : '未命名对话',
    messageCount: messages.length,
    createdAt: typeof session.createdAt === 'string'
      ? session.createdAt.slice(0, 64)
      : new Date(0).toISOString(),
    updatedAt: typeof session.updatedAt === 'string'
      ? session.updatedAt.slice(0, 64)
      : new Date(0).toISOString(),
    messages
  }
}

export const loadSessions = (storage, key) => {
  try {
    const raw = storage.getItem(key)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed
      .slice(0, MAX_STORED_SESSIONS)
      .map(normalizeSession)
      .filter(Boolean)
  } catch (error) {
    console.warn('Unable to load chat sessions:', error)
    return []
  }
}

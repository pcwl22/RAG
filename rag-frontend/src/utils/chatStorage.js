export const MAX_STORED_SESSIONS = 20
export const MAX_STORED_MESSAGES = 100
const MAX_CONTENT_LENGTH = 20000

const compactMessage = (message = {}) => ({
  role: message.role,
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

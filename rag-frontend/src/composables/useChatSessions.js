import { computed, nextTick, ref } from 'vue'
import { normalizeProcessTrace } from '../utils/processTrace'
import { saveSessions } from '../utils/chatStorage'

const STORAGE_KEY = 'rag_chat_sessions'

const normalizeStoredMessages = (items = []) =>
  items.map(msg => {
    if (msg.role !== 'assistant') return msg
    return {
      ...msg,
      streaming: false,
      streamingStatus: msg.content ? '' : msg.streamingStatus || '',
      showUnderstanding: false,
      process: normalizeProcessTrace(msg.process)
    }
  })

const createSession = (index) => ({
  id: Date.now().toString(),
  title: `对话 ${index + 1}`,
  messages: [],
  messageCount: 0,
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString()
})

export const useChatSessions = ({ onAfterSwitch } = {}) => {
  const chatSessions = ref([])
  const currentSessionId = ref(null)
  const messages = ref([])

  const currentChatTitle = computed(() => {
    const session = chatSessions.value.find(s => s.id === currentSessionId.value)
    return session?.title || '新对话'
  })

  const saveChats = () => {
    saveSessions(localStorage, STORAGE_KEY, chatSessions.value)
  }

  const createNewChat = () => {
    const newSession = createSession(chatSessions.value.length)
    chatSessions.value.unshift(newSession)
    currentSessionId.value = newSession.id
    messages.value = []
    saveChats()
  }

  const switchChat = (sessionId) => {
    const session = chatSessions.value.find(s => s.id === sessionId)
    if (!session) return

    currentSessionId.value = sessionId
    session.messages = normalizeStoredMessages(session.messages || [])
    messages.value = session.messages
    nextTick(() => onAfterSwitch?.())
  }

  const deleteChat = (sessionId) => {
    if (!confirm('确定要删除这个对话吗？')) return

    chatSessions.value = chatSessions.value.filter(s => s.id !== sessionId)

    if (currentSessionId.value === sessionId) {
      if (chatSessions.value.length > 0) {
        switchChat(chatSessions.value[0].id)
      } else {
        createNewChat()
      }
    }
    saveChats()
  }

  const updateCurrentSession = () => {
    const session = chatSessions.value.find(s => s.id === currentSessionId.value)
    if (!session) return

    session.messages = messages.value
    session.messageCount = messages.value.length
    session.updatedAt = new Date().toISOString()

    if (session.messageCount > 0 && session.title.startsWith('对话')) {
      const firstUserMsg = messages.value.find(m => m.role === 'user')
      if (firstUserMsg) {
        session.title = firstUserMsg.content.substring(0, 20) + (firstUserMsg.content.length > 20 ? '...' : '')
      }
    }

    saveChats()
  }

  const initChats = () => {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (!saved) {
      createNewChat()
      return
    }

    try {
      const parsed = JSON.parse(saved)
      if (!Array.isArray(parsed)) throw new TypeError('Stored chat sessions must be an array')
      chatSessions.value = parsed
      chatSessions.value.forEach(session => {
        session.messages = normalizeStoredMessages(session.messages || [])
      })
      if (chatSessions.value.length > 0) {
        currentSessionId.value = chatSessions.value[0].id
        messages.value = chatSessions.value[0].messages
      } else {
        createNewChat()
      }
    } catch (error) {
      console.error('Failed to load chat sessions:', error)
      createNewChat()
    }
  }

  return {
    chatSessions,
    currentSessionId,
    messages,
    currentChatTitle,
    createNewChat,
    deleteChat,
    initChats,
    saveChats,
    switchChat,
    updateCurrentSession
  }
}

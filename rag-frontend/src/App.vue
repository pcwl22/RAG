<template>
  <div class="rag-chat">
    <div class="header">
      <h1>🤖 RAG智能问答系统</h1>
      <div class="status">
        <span v-if="oidcEnabled && authUserName" class="auth-user">{{ authUserName }}</span>
        <button v-if="oidcEnabled" class="btn-tool" @click="handleSignOut">退出登录</button>
        <span :class="{ online: isOnline }">{{ isOnline ? '在线' : '离线' }}</span>
      </div>
    </div>

    <div class="main-container">
      <ChatSidebar
        :chat-sessions="chatSessions"
        :current-session-id="currentSessionId"
        @new-chat="createNewChat"
        @switch-chat="switchChat"
        @delete-chat="deleteChat"
      />

      <!-- 主对话区域 -->
      <main class="chat-area">
        <!-- 工具栏 -->
        <div class="chat-toolbar">
          <div class="toolbar-left">
            <span class="current-chat-title">{{ currentChatTitle }}</span>
          </div>
          <div class="toolbar-right">
            <button @click="showUpload = true" class="btn-tool" title="上传文档">
              📤 上传文档
            </button>
            <button @click="showDocuments = !showDocuments" class="btn-tool" title="文档库">
              📚 文档库
            </button>
          </div>
        </div>

        <div class="messages-container" ref="messagesContainer">
          <div v-if="messages.length === 0" class="welcome">
            <h2>👋 欢迎使用RAG智能问答</h2>
            <p>上传文档并开始提问吧</p>
          </div>

          <div
            v-for="(msg, index) in messages"
            :key="index"
            class="message"
            :class="msg.role"
          >
            <div class="message-avatar">
              {{ msg.role === 'user' ? '👤' : '🤖' }}
            </div>
            <div class="message-content">
              <div
                v-if="msg.content || msg.streamingStatus"
                class="message-text"
                :class="{ streaming: msg.streaming }"
              >
                <span v-if="msg.content">{{ msg.content }}</span>
                <span v-else class="stream-status">{{ msg.streamingStatus }}</span>
                <span v-if="msg.streaming && msg.content" class="stream-cursor"></span>
              </div>

              <ProcessTrace
                v-if="msg.role === 'assistant' && msg.process"
                :process="msg.process"
                :context-count="msg.contexts?.length || 0"
                @toggle="msg.process.show = !msg.process.show"
              />

              <RetrievalContexts
                v-if="msg.contexts && msg.contexts.length > 0"
                :contexts="msg.contexts"
                :show="msg.showContexts"
                @toggle="msg.showContexts = !msg.showContexts"
              />

              <QueryUnderstandingPanel
                v-if="msg.understanding"
                :understanding="msg.understanding"
                :show="msg.showUnderstanding"
                @toggle="msg.showUnderstanding = !msg.showUnderstanding"
              />
            </div>
          </div>

          <div v-if="isLoading && !hasActiveStreamingMessage" class="message assistant">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
              <div class="loading">思考中...</div>
            </div>
          </div>
        </div>

        <!-- 输入区域 -->
        <div class="input-area">
          <div class="input-options">
            <label>
              <input type="checkbox" v-model="options.enableUnderstanding">
              启用查询理解
            </label>
            <label>
              <input type="checkbox" v-model="options.stream">
              流式输出
            </label>
          </div>
          <div class="input-row">
            <textarea
              v-model="userInput"
              @keydown.enter.exact.prevent="handleSend"
              placeholder="输入你的问题... (Enter发送, Shift+Enter换行)"
              rows="3"
            ></textarea>
            <button
              @click="handleSend"
              :disabled="!userInput.trim() || isLoading"
              class="btn-send"
            >
              发送
            </button>
          </div>
        </div>
      </main>
    </div>

    <UploadModal
      v-if="showUpload"
      :selected-file="selectedFile"
      :upload-partition="uploadPartition"
      :upload-progress="uploadProgress"
      :uploading="uploading"
      @close="showUpload = false"
      @update:selected-file="selectedFile = $event"
      @update:upload-partition="uploadPartition = $event"
      @upload="handleUpload"
    />

    <DocumentsModal
      v-if="showDocuments"
      :documents="documents"
      :selected-doc="selectedDoc"
      :selected-doc-chunks="selectedDocChunks"
      :selected-doc-chunk-total="selectedDocChunkTotal"
      :document-chunks-loading="documentChunksLoading"
      :document-chunks-error="documentChunksError"
      @close="showDocuments = false"
      @load-chunks="loadDocumentChunks"
      @toggle-chunk="toggleDocumentChunk"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import ChatSidebar from './components/ChatSidebar.vue'
import DocumentsModal from './components/DocumentsModal.vue'
import ProcessTrace from './components/ProcessTrace.vue'
import QueryUnderstandingPanel from './components/QueryUnderstandingPanel.vue'
import RetrievalContexts from './components/RetrievalContexts.vue'
import UploadModal from './components/UploadModal.vue'
import {
  ENDPOINTS,
  fetchDocumentChunks,
  fetchDocumentStatus,
  fetchDocuments,
  fetchHealth,
  ingestDocument,
  postJson,
  postStream
} from './api/ragApi'
import {
  addProcessEvent,
  addProcessStep,
  createProcessTrace,
  finishProcess
} from './utils/processTrace'
import {
  createStreamWriter,
  readEventStream
} from './utils/eventStream'
import { useChatSessions } from './composables/useChatSessions'
import { getAuthenticatedUser, oidcEnabled, signOut } from './auth/oidc.js'
import { buildChatStorageKey } from './utils/chatStorage.js'

const isOnline = ref(false)
const authUser = getAuthenticatedUser()
const authUserName = authUser?.profile?.preferred_username || authUser?.profile?.name || ''
const chatStorageKey = buildChatStorageKey(authUser)
const documents = ref([])
const selectedDoc = ref(null)
const selectedDocChunks = ref([])
const selectedDocChunkTotal = ref(0)
const documentChunksLoading = ref(false)
const documentChunksError = ref('')
const showDocuments = ref(false)

// 对话管理
const {
  chatSessions,
  currentSessionId,
  messages,
  currentChatTitle,
  createNewChat,
  deleteChat,
  initChats,
  switchChat,
  updateCurrentSession
} = useChatSessions({
  onAfterSwitch: () => scrollToBottom(),
  storageKey: chatStorageKey
})

const handleSignOut = async () => {
  try {
    localStorage.removeItem(chatStorageKey)
  } finally {
    await signOut()
  }
}

const userInput = ref('')
const isLoading = ref(false)
const messagesContainer = ref(null)
const activeRequestController = ref(null)
let healthIntervalId = null

const showUpload = ref(false)
const selectedFile = ref(null)
const uploadPartition = ref('general')
const uploading = ref(false)
const uploadProgress = ref(0)

const options = ref({
  enableUnderstanding: true,
  stream: true
})

const UPLOAD_POLL_INTERVAL_MS = 1000
const UPLOAD_MAX_POLLS = 300
const delay = (milliseconds) => new Promise(resolve => setTimeout(resolve, milliseconds))

const hasActiveStreamingMessage = computed(() =>
  messages.value.some(msg => msg.role === 'assistant' && msg.streaming)
)

// 检查系统状态
const checkHealth = async () => {
  isOnline.value = await fetchHealth()
}

// 加载文档列表
const loadDocuments = async () => {
  try {
    const data = await fetchDocuments()
    documents.value = data.documents || []
    if (selectedDoc.value) {
      const refreshed = documents.value.find(doc => doc.document_id === selectedDoc.value.document_id)
      if (refreshed) {
        selectedDoc.value = refreshed
      } else {
        selectedDoc.value = null
        selectedDocChunks.value = []
        selectedDocChunkTotal.value = 0
      }
    }
  } catch (error) {
    console.error('加载文档失败:', error)
  }
}

const loadDocumentChunks = async (doc) => {
  if (!doc?.document_id) return
  selectedDoc.value = doc
  documentChunksLoading.value = true
  documentChunksError.value = ''

  try {
    const data = await fetchDocumentChunks(doc.document_id, { limit: 5000 })
    selectedDocChunkTotal.value = data.total || 0
    selectedDocChunks.value = (data.chunks || []).map((chunk, index) => ({
      ...chunk,
      show: index === 0
    }))
  } catch (error) {
    selectedDocChunks.value = []
    selectedDocChunkTotal.value = 0
    documentChunksError.value = `加载切片失败: ${error.message}`
  } finally {
    documentChunksLoading.value = false
  }
}

const toggleDocumentChunk = (chunk) => {
  if (!chunk) return
  chunk.show = !chunk.show
}

// 发送消息
const handleSend = async () => {
  if (!userInput.value.trim() || isLoading.value) return

  const query = userInput.value.trim()
  userInput.value = ''
  const originSessionId = currentSessionId.value
  const originMessages = messages.value
  const scrollOriginToBottom = () => {
    if (currentSessionId.value === originSessionId) scrollToBottom()
  }
  const chatHistory = originMessages.slice(-6).map(m => ({
    role: m.role,
    content: m.content
  }))

  // 添加用户消息
  originMessages.push({
    role: 'user',
    content: query,
    originalQuery: query  // 保存原始查询供后续使用
  })

  isLoading.value = true
  scrollOriginToBottom()

  let streamingMessage = null
  let streamWriter = null
  let activeProcessTrace = null
  let requestController = null

  try {
    if (options.value.stream) {
      requestController = new AbortController()
      activeRequestController.value = requestController
      const endpoint = options.value.enableUnderstanding ? ENDPOINTS.enhancedQuery : ENDPOINTS.answer
      const payload = options.value.enableUnderstanding
        ? {
            query,
            chat_history: chatHistory,
            stream: true
          }
        : { query, stream: true }
      const processTrace = createProcessTrace({
        mode: options.value.enableUnderstanding ? '增强查询 / 流式' : '标准问答 / 流式',
        endpoint,
        request: payload
      })
      processTrace.streaming = true
      activeProcessTrace = processTrace
      addProcessStep(processTrace, '发送请求', endpoint)

      const assistantMessage = {
        role: 'assistant',
        content: '',
        contexts: [],
        highConfidenceContexts: [],
        understanding: null,
        process: processTrace,
        streamingStatus: '正在连接...',
        streaming: true,
        showContexts: false,
        showUnderstanding: false,
        query: query
      }
      originMessages.push(assistantMessage)
      streamingMessage = originMessages[originMessages.length - 1]
      streamWriter = createStreamWriter((text) => {
        streamingMessage.streamingStatus = ''
        streamingMessage.content += text
        scrollOriginToBottom()
      })

      const res = await postStream(endpoint, payload, { signal: requestController.signal })
      await readEventStream(res, {
        onEvent: (payload) => {
          addProcessEvent(streamingMessage.process, payload)
        },
        onChunk: (chunk) => {
          streamWriter.enqueue(chunk)
        },
        onSources: (sources) => {
          streamingMessage.contexts = sources || []
          streamingMessage.highConfidenceContexts = streamingMessage.contexts.filter(ctx => ctx.score >= 0.8)
          addProcessStep(streamingMessage.process, '收到检索结果', `${streamingMessage.contexts.length} 条`)
        },
        onUnderstanding: (understanding) => {
          streamingMessage.understanding = understanding
          addProcessStep(
            streamingMessage.process,
            '查询理解完成',
            `${understanding?.retrieval_queries?.length || 0} 个实际检索查询`
          )
        },
        onStatus: (status) => {
          if (!streamingMessage.content) {
            streamingMessage.streamingStatus = status
          }
          addProcessStep(streamingMessage.process, '后端状态', status)
        },
        onDone: (payload) => {
          finishProcess(streamingMessage.process, payload?.total_time ?? null)
        }
      })
      await streamWriter.drain()
      streamingMessage.streaming = false
      streamingMessage.process.streaming = false

      return
    }

    let response

    if (options.value.enableUnderstanding) {
      // 使用增强查询
      const endpoint = ENDPOINTS.enhancedQuery
      const payload = {
        query,
        chat_history: chatHistory
      }
      const processTrace = createProcessTrace({
        mode: '增强查询 / 非流式',
        endpoint,
        request: payload
      })
      activeProcessTrace = processTrace
      addProcessStep(processTrace, '发送请求', endpoint)

      const data = await postJson(endpoint, payload)
      addProcessEvent(processTrace, {
        type: 'json_response',
        data: {
          result_count: data.results?.length || 0,
          has_understanding: Boolean(data.understanding),
          sub_answer_count: data.sub_answers?.length || 0,
          answer_length: data.answer?.length || 0
        }
      })
      addProcessStep(processTrace, '收到响应', `${data.results?.length || 0} 条检索结果`)
      processTrace.subAnswers = data.sub_answers || []
      finishProcess(processTrace)
      response = {
        content: data.answer,
        contexts: data.results,
        understanding: data.understanding,
        process: processTrace
      }
    } else {
      // 标准查询
      const endpoint = ENDPOINTS.answer
      const payload = { query }
      const processTrace = createProcessTrace({
        mode: '标准问答 / 非流式',
        endpoint,
        request: payload
      })
      activeProcessTrace = processTrace
      addProcessStep(processTrace, '发送请求', endpoint)

      const data = await postJson(endpoint, payload)
      const contexts = data.sources || data.contexts || []
      addProcessEvent(processTrace, {
        type: 'json_response',
        data: {
          result_count: contexts.length,
          answer_length: data.answer?.length || 0,
          total_time: data.total_time
        }
      })
      addProcessStep(processTrace, '收到响应', `${contexts.length} 条检索结果`)
      finishProcess(processTrace, data.total_time ?? null)
      response = {
        content: data.answer,
        contexts,
        process: processTrace
      }
    }

    // 过滤置信度≥0.8的文档
    const highConfidenceContexts = response.contexts?.filter(ctx => ctx.score >= 0.8) || []

    originMessages.push({
      role: 'assistant',
      content: response.content,
      contexts: response.contexts,
      highConfidenceContexts: highConfidenceContexts,
      understanding: response.understanding,
      process: response.process,
      showContexts: false,
      showUnderstanding: false,
      query: query  // 保存查询用于提取相关内容
    })

  } catch (error) {
    if (streamingMessage) {
      streamingMessage.process.error = error.message
      addProcessStep(streamingMessage.process, '流程异常', error.message)
      addProcessEvent(streamingMessage.process, { type: 'error', error: error.message })
      try {
        await streamWriter?.drain()
      } catch (drainError) {
        console.error('Failed to drain stream writer:', drainError)
      }

      streamingMessage.streaming = false
      streamingMessage.process.streaming = false
      streamingMessage.streamingStatus = ''
      if (streamingMessage.content) {
        streamingMessage.content += `\n\n[流式输出中断：${error.message}]`
      } else {
        streamingMessage.content = '抱歉，查询失败: ' + error.message
      }
    } else {
      if (activeProcessTrace) {
        activeProcessTrace.error = error.message
        addProcessStep(activeProcessTrace, '流程异常', error.message)
        addProcessEvent(activeProcessTrace, { type: 'error', error: error.message })
        finishProcess(activeProcessTrace)
      }

      const lastMessage = originMessages[originMessages.length - 1]
      if (lastMessage?.role === 'assistant' && !lastMessage.content) {
        lastMessage.content = '抱歉，查询失败: ' + error.message
        lastMessage.process = activeProcessTrace
      } else {
        originMessages.push({
          role: 'assistant',
          content: '抱歉，查询失败: ' + error.message,
          contexts: [],
          highConfidenceContexts: [],
          process: activeProcessTrace,
          showContexts: false,
          query
        })
      }
    }
  } finally {
    if (activeRequestController.value === requestController) {
      activeRequestController.value = null
    }
    isLoading.value = false
    updateCurrentSession(originSessionId, originMessages)
    scrollOriginToBottom()
  }
}

// 上传文档
const handleUpload = async () => {
  if (!selectedFile.value) return

  uploading.value = true
  uploadProgress.value = 0

  try {
    const submitted = await ingestDocument(selectedFile.value, uploadPartition.value)
    if (!submitted?.task_id) throw new Error('上传接口未返回任务 ID')

    uploadProgress.value = 10
    let completed = false
    for (let attempt = 0; attempt < UPLOAD_MAX_POLLS; attempt += 1) {
      const status = await fetchDocumentStatus(submitted.task_id)
      if (status.status === 'completed') {
        uploadProgress.value = 100
        completed = true
        break
      }
      if (status.status === 'failed') {
        throw new Error(status.error || '文档处理失败')
      }
      uploadProgress.value = Math.max(10, Math.min(95, status.progress || 50))
      await delay(UPLOAD_POLL_INTERVAL_MS)
    }
    if (!completed) throw new Error('文档处理超时，请稍后在文档库中确认任务状态')

    await loadDocuments()
    showUpload.value = false
    selectedFile.value = null
    uploadProgress.value = 0
  } catch (error) {
    alert('上传失败: ' + error.message)
  } finally {
    uploading.value = false
  }
}

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

onMounted(() => {
  initChats()  // 初始化对话历史
  checkHealth()
  loadDocuments()
  healthIntervalId = setInterval(checkHealth, 30000)
})

onUnmounted(() => {
  activeRequestController.value?.abort()
  activeRequestController.value = null
  if (healthIntervalId !== null) clearInterval(healthIntervalId)
})
</script>

<style scoped>
.rag-chat {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #f7f5f1;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.header {
  background: white;
  padding: 1rem 2rem;
  border-bottom: 1px solid #e7e3da;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header h1 {
  margin: 0;
  font-size: 1.5rem;
  color: #1e2227;
}

.status span {
  padding: 0.25rem 0.75rem;
  border-radius: 1rem;
  font-size: 0.875rem;
  background: #e7e3da;
  color: #6b7077;
}

.status span.online {
  background: #d1fae5;
  color: #065f46;
}

.main-container {
  flex: 1;
  display: flex;
  overflow: hidden;
}

.chat-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: white;
}

.chat-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  border-bottom: 1px solid #e7e3da;
  background: white;
}

.toolbar-left {
  flex: 1;
}

.current-chat-title {
  font-weight: 500;
  color: #1e2227;
}

.toolbar-right {
  display: flex;
  gap: 0.5rem;
}

.btn-tool {
  padding: 0.5rem 1rem;
  background: white;
  border: 1px solid #e7e3da;
  border-radius: 0.5rem;
  cursor: pointer;
  font-size: 0.875rem;
  transition: all 0.2s;
}

.btn-tool:hover {
  background: #f7f5f1;
  border-color: #3c5a78;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 2rem;
}

.welcome {
  text-align: center;
  padding: 4rem 2rem;
  color: #6b7077;
}

.message {
  display: flex;
  margin-bottom: 1.5rem;
}

.message-avatar {
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  flex-shrink: 0;
  margin-right: 1rem;
}

.message.user .message-avatar {
  background: #3c5a78;
}

.message.assistant .message-avatar {
  background: #e7e3da;
}

.message-content {
  flex: 1;
  min-width: 0;
}

.message-text {
  background: #f7f5f1;
  padding: 1rem;
  border-radius: 0.75rem;
  line-height: 1.6;
  white-space: pre-wrap;
}

.message-text.streaming {
  border: 1px solid #d8e1eb;
}

.message.user .message-text {
  background: #3c5a78;
  color: white;
}

.loading {
  padding: 1rem;
  color: #6b7077;
  font-style: italic;
}

.stream-status {
  color: #6b7077;
  font-size: 0.9rem;
  font-style: italic;
}

.stream-cursor {
  display: inline-block;
  width: 0.55em;
  height: 1.05em;
  margin-left: 0.12em;
  vertical-align: -0.16em;
  background: #3c5a78;
  animation: stream-cursor-blink 1s steps(2, start) infinite;
}

@keyframes stream-cursor-blink {
  50% {
    opacity: 0;
  }
}

.input-area {
  border-top: 1px solid #e7e3da;
  padding: 1rem 2rem;
  background: white;
}

.input-options {
  display: flex;
  gap: 1.5rem;
  margin-bottom: 0.75rem;
  font-size: 0.875rem;
}

.input-options label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
}

.input-row {
  display: flex;
  gap: 1rem;
}

textarea {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #e7e3da;
  border-radius: 0.5rem;
  resize: none;
  font-family: inherit;
  font-size: 1rem;
}

.btn-send {
  padding: 0.75rem 2rem;
  background: #3c5a78;
  color: white;
  border: none;
  border-radius: 0.5rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-send:hover:not(:disabled) {
  background: #2e4760;
}

.btn-send:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>

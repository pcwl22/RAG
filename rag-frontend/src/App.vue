<template>
  <div class="rag-chat">
    <div class="header">
      <h1>🤖 RAG智能问答系统</h1>
      <div class="status">
        <span :class="{ online: isOnline }">{{ isOnline ? '在线' : '离线' }}</span>
      </div>
    </div>

    <div class="main-container">
      <!-- 左侧边栏 - 对话历史 -->
      <aside class="sidebar sidebar-left">
        <div class="sidebar-header">
          <h2>💬 对话历史</h2>
          <button @click="createNewChat" class="btn-primary">新建对话</button>
        </div>

        <div class="chat-list">
          <div v-if="chatSessions.length === 0" class="empty-state">
            暂无对话
          </div>
          <div
            v-for="session in chatSessions"
            :key="session.id"
            class="chat-item"
            :class="{ active: currentSessionId === session.id }"
            @click="switchChat(session.id)"
          >
            <div class="chat-icon">💬</div>
            <div class="chat-info">
              <div class="chat-title">{{ session.title }}</div>
              <div class="chat-meta">{{ session.messageCount }} 条消息 · {{ formatTime(session.updatedAt) }}</div>
            </div>
            <button
              v-if="chatSessions.length > 1"
              @click.stop="deleteChat(session.id)"
              class="btn-delete"
              title="删除对话"
            >✕</button>
          </div>
        </div>
      </aside>

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
              <div class="message-text">{{ msg.content }}</div>

              <!-- 显示检索上下文 -->
              <div v-if="msg.highConfidenceContexts && msg.highConfidenceContexts.length > 0" class="contexts">
                <div class="contexts-header" @click="msg.showContexts = !msg.showContexts">
                  📑 检索到 {{ msg.highConfidenceContexts.length }} 个高置信度文档 (≥0.8)
                  <span class="toggle">{{ msg.showContexts ? '▼' : '▶' }}</span>
                </div>
                <div v-show="msg.showContexts" class="contexts-list">
                  <div v-for="(ctx, idx) in msg.highConfidenceContexts" :key="idx" class="context-item">
                    <div class="context-score">得分: {{ ctx.score?.toFixed(3) }}</div>
                    <div class="context-text">{{ getRelevantSnippet(ctx.content, msg.query) }}</div>
                  </div>
                </div>
              </div>

              <!-- 显示查询理解 -->
              <div v-if="msg.understanding" class="understanding">
                <div class="understanding-header">🧠 查询理解过程</div>

                <div class="understanding-step">
                  <strong>原始查询：</strong>
                  <div class="step-content">{{ msg.understanding.original_query }}</div>
                </div>

                <div v-if="msg.understanding.resolved_query !== msg.understanding.original_query" class="understanding-step">
                  <strong>✓ 指代消解：</strong>
                  <div class="step-content highlight">{{ msg.understanding.resolved_query }}</div>
                  <div class="step-note">将代词还原为明确实体</div>
                </div>

                <div v-if="msg.understanding.subqueries && msg.understanding.subqueries.length > 1" class="understanding-step">
                  <strong>✓ 查询拆分：</strong>
                  <div class="step-note">复合问题拆分为 {{ msg.understanding.subqueries.length }} 个子查询</div>
                  <ol class="subqueries-list">
                    <li v-for="(sq, i) in msg.understanding.subqueries" :key="i">{{ sq }}</li>
                  </ol>
                </div>

                <div v-if="!msg.understanding.is_decomposed && msg.understanding.resolved_query === msg.understanding.original_query" class="understanding-step">
                  <strong>✓ 分析结果：</strong>
                  <div class="step-content">单一问题，无需拆分或改写</div>
                </div>
              </div>
            </div>
          </div>

          <div v-if="isLoading" class="message assistant">
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
              <input type="checkbox" v-model="options.enableEvaluation">
              启用评估
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

    <!-- 文档上传对话框 -->
    <div v-if="showUpload" class="modal" @click.self="showUpload = false">
      <div class="modal-content">
        <div class="modal-header">
          <h3>📤 上传文档</h3>
          <button @click="showUpload = false" class="btn-close">✕</button>
        </div>
        <div class="modal-body">
          <div class="upload-area" @drop.prevent="handleDrop" @dragover.prevent>
            <input
              type="file"
              ref="fileInput"
              @change="handleFileSelect"
              accept=".pdf,.txt,.md,.docx,.xlsx"
              style="display: none"
            >
            <div class="upload-prompt" @click="$refs.fileInput.click()">
              <div class="upload-icon">📁</div>
              <div>点击选择文件或拖拽到此处</div>
              <div class="upload-hint">支持 PDF, TXT, MD, DOCX, XLSX</div>
            </div>
          </div>

          <div v-if="selectedFile" class="selected-file">
            <div>已选择: {{ selectedFile.name }}</div>
            <select v-model="uploadPartition">
              <option value="general">通用</option>
              <option value="contract">合同</option>
              <option value="manual">手册</option>
              <option value="process">流程</option>
              <option value="policy">制度</option>
            </select>
          </div>

          <div v-if="uploadProgress > 0" class="progress-bar">
            <div class="progress-fill" :style="{ width: uploadProgress + '%' }"></div>
          </div>
        </div>
        <div class="modal-footer">
          <button @click="showUpload = false" class="btn-secondary">取消</button>
          <button
            @click="handleUpload"
            :disabled="!selectedFile || uploading"
            class="btn-primary"
          >
            {{ uploading ? '上传中...' : '上传' }}
          </button>
        </div>
      </div>
    </div>

    <!-- 文档库对话框 -->
    <div v-if="showDocuments" class="modal" @click.self="showDocuments = false">
      <div class="modal-content">
        <div class="modal-header">
          <h3>📚 文档库</h3>
          <button @click="showDocuments = false" class="btn-close">✕</button>
        </div>
        <div class="modal-body">
          <div v-if="documents.length === 0" class="empty-state">
            暂无文档
          </div>
          <div v-else class="documents-grid">
            <div
              v-for="doc in documents"
              :key="doc.document_id"
              class="document-card"
            >
              <div class="doc-icon">📄</div>
              <div class="doc-info">
                <div class="doc-name">{{ doc.filename }}</div>
                <div class="doc-meta">
                  {{ doc.chunk_count }} 块 · {{ doc.partition }}
                </div>
                <div class="doc-date">{{ formatTime(doc.updated_at) }}</div>
              </div>
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button @click="showDocuments = false" class="btn-secondary">关闭</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick } from 'vue'

const API_BASE = 'http://localhost:8000/api/v1'

const isOnline = ref(false)
const documents = ref([])
const selectedDoc = ref(null)
const showDocuments = ref(false)

// 对话管理
const chatSessions = ref([])
const currentSessionId = ref(null)
const messages = ref([])

const userInput = ref('')
const isLoading = ref(false)
const messagesContainer = ref(null)

const showUpload = ref(false)
const selectedFile = ref(null)
const uploadPartition = ref('general')
const uploading = ref(false)
const uploadProgress = ref(0)

const options = ref({
  enableUnderstanding: true,
  enableEvaluation: false
})

// 计算属性：当前对话标题
const currentChatTitle = computed(() => {
  const session = chatSessions.value.find(s => s.id === currentSessionId.value)
  return session?.title || '新对话'
})

// 初始化对话
const initChats = () => {
  const saved = localStorage.getItem('rag_chat_sessions')
  if (saved) {
    try {
      chatSessions.value = JSON.parse(saved)
      if (chatSessions.value.length > 0) {
        currentSessionId.value = chatSessions.value[0].id
        messages.value = chatSessions.value[0].messages || []
      } else {
        createNewChat()
      }
    } catch (e) {
      console.error('Failed to load chat sessions:', e)
      createNewChat()
    }
  } else {
    createNewChat()
  }
}

// 保存对话到本地存储
const saveChats = () => {
  localStorage.setItem('rag_chat_sessions', JSON.stringify(chatSessions.value))
}

// 新建对话
const createNewChat = () => {
  const newSession = {
    id: Date.now().toString(),
    title: `对话 ${chatSessions.value.length + 1}`,
    messages: [],
    messageCount: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  }
  chatSessions.value.unshift(newSession)
  currentSessionId.value = newSession.id
  messages.value = []
  saveChats()
}

// 切换对话
const switchChat = (sessionId) => {
  const session = chatSessions.value.find(s => s.id === sessionId)
  if (session) {
    currentSessionId.value = sessionId
    messages.value = session.messages || []
    nextTick(() => scrollToBottom())
  }
}

// 删除对话
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

// 更新当前对话
const updateCurrentSession = () => {
  const session = chatSessions.value.find(s => s.id === currentSessionId.value)
  if (session) {
    session.messages = messages.value
    session.messageCount = messages.value.length
    session.updatedAt = new Date().toISOString()

    // 自动生成标题（使用第一条用户消息）
    if (session.messageCount > 0 && session.title.startsWith('对话')) {
      const firstUserMsg = messages.value.find(m => m.role === 'user')
      if (firstUserMsg) {
        session.title = firstUserMsg.content.substring(0, 20) + (firstUserMsg.content.length > 20 ? '...' : '')
      }
    }

    saveChats()
  }
}

// 格式化时间
const formatTime = (dateStr) => {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  const now = new Date()
  const diff = now - date

  if (diff < 60000) return '刚刚'
  if (diff < 3600000) return Math.floor(diff / 60000) + '分钟前'
  if (diff < 86400000) return Math.floor(diff / 3600000) + '小时前'
  if (diff < 604800000) return Math.floor(diff / 86400000) + '天前'

  return date.toLocaleDateString()
}

// 检查系统状态
const checkHealth = async () => {
  try {
    const res = await fetch(`http://localhost:8000/health`)
    isOnline.value = res.ok
  } catch (error) {
    isOnline.value = false
  }
}

// 加载文档列表
const loadDocuments = async () => {
  try {
    const res = await fetch(`${API_BASE}/documents`)
    const data = await res.json()
    documents.value = data.documents || []
  } catch (error) {
    console.error('加载文档失败:', error)
  }
}

// 发送消息
const handleSend = async () => {
  if (!userInput.value.trim() || isLoading.value) return

  const query = userInput.value.trim()
  userInput.value = ''

  // 添加用户消息
  messages.value.push({
    role: 'user',
    content: query,
    originalQuery: query  // 保存原始查询供后续使用
  })

  isLoading.value = true
  scrollToBottom()

  try {
    let response

    if (options.value.enableUnderstanding) {
      // 使用增强查询
      const res = await fetch(`${API_BASE}/query/enhanced`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          chat_history: messages.value.slice(-6).map(m => ({
            role: m.role,
            content: m.content
          })),
          enable_evaluation: options.value.enableEvaluation,
          top_k: 5
        })
      })

      const data = await res.json()
      response = {
        content: data.answer,
        contexts: data.results,
        understanding: data.understanding,
        evaluation: data.evaluation
      }
    } else {
      // 标准查询
      const res = await fetch(`${API_BASE}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 })
      })

      const data = await res.json()
      response = {
        content: data.answer,
        contexts: data.contexts
      }
    }

    // 过滤置信度≥0.8的文档
    const highConfidenceContexts = response.contexts?.filter(ctx => ctx.score >= 0.8) || []

    messages.value.push({
      role: 'assistant',
      content: response.content,
      contexts: response.contexts,
      highConfidenceContexts: highConfidenceContexts,
      understanding: response.understanding,
      evaluation: response.evaluation,
      showContexts: false,
      query: query  // 保存查询用于提取相关内容
    })

  } catch (error) {
    messages.value.push({
      role: 'assistant',
      content: '抱歉，查询失败: ' + error.message
    })
  } finally {
    isLoading.value = false
    updateCurrentSession()  // 保存对话
    scrollToBottom()
  }
}

// 文件选择
const handleFileSelect = (e) => {
  selectedFile.value = e.target.files[0]
}

const handleDrop = (e) => {
  selectedFile.value = e.dataTransfer.files[0]
}

// 上传文档
const handleUpload = async () => {
  if (!selectedFile.value) return

  uploading.value = true
  uploadProgress.value = 0

  const formData = new FormData()
  formData.append('file', selectedFile.value)
  formData.append('partition', uploadPartition.value)

  try {
    const res = await fetch(`${API_BASE}/documents/ingest`, {
      method: 'POST',
      body: formData
    })

    if (res.ok) {
      uploadProgress.value = 100
      setTimeout(() => {
        showUpload.value = false
        selectedFile.value = null
        uploadProgress.value = 0
        loadDocuments()
      }, 500)
    } else {
      alert('上传失败')
    }
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

// 提取相关摘要（智能截取包含关键词的部分）
const getRelevantSnippet = (content, query) => {
  if (!content || !query) return content?.substring(0, 200) + '...'

  // 提取查询中的关键词
  const keywords = query.replace(/[？?！!，,。.]/g, ' ').split(' ').filter(w => w.length > 1)

  // 查找第一个匹配的关键词位置
  let matchIndex = -1
  for (const keyword of keywords) {
    matchIndex = content.indexOf(keyword)
    if (matchIndex !== -1) break
  }

  // 如果找到关键词，从该位置前后截取
  if (matchIndex !== -1) {
    const start = Math.max(0, matchIndex - 50)
    const end = Math.min(content.length, matchIndex + 300)
    let snippet = content.substring(start, end)

    if (start > 0) snippet = '...' + snippet
    if (end < content.length) snippet = snippet + '...'

    return snippet
  }

  // 如果没找到关键词，返回前200字符
  return content.substring(0, 200) + (content.length > 200 ? '...' : '')
}

onMounted(() => {
  initChats()  // 初始化对话历史
  checkHealth()
  loadDocuments()
  setInterval(checkHealth, 30000)
})
</script>

<style scoped>
* {
  box-sizing: border-box;
}

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

.sidebar {
  width: 280px;
  background: white;
  border-right: 1px solid #e7e3da;
  display: flex;
  flex-direction: column;
}

.sidebar-left {
  border-right: 1px solid #e7e3da;
}

.sidebar-header {
  padding: 1.5rem;
  border-bottom: 1px solid #e7e3da;
}

.sidebar-header h2 {
  margin: 0 0 1rem 0;
  font-size: 1.125rem;
  color: #1e2227;
}

.chat-list {
  flex: 1;
  overflow-y: auto;
  padding: 0.5rem;
}

.chat-item {
  display: flex;
  align-items: center;
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  border-radius: 0.5rem;
  cursor: pointer;
  transition: background 0.2s;
  position: relative;
}

.chat-item:hover {
  background: #f7f5f1;
}

.chat-item:hover .btn-delete {
  display: block;
}

.chat-item.active {
  background: #3c5a78;
  color: white;
}

.chat-icon {
  font-size: 1.5rem;
  margin-right: 0.75rem;
}

.chat-info {
  flex: 1;
  min-width: 0;
}

.chat-title {
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chat-meta {
  font-size: 0.75rem;
  opacity: 0.7;
  margin-top: 0.25rem;
}

.btn-delete {
  display: none;
  position: absolute;
  right: 0.5rem;
  top: 50%;
  transform: translateY(-50%);
  background: #ef4444;
  color: white;
  border: none;
  border-radius: 50%;
  width: 1.5rem;
  height: 1.5rem;
  cursor: pointer;
  font-size: 0.875rem;
  line-height: 1;
}

.btn-delete:hover {
  background: #dc2626;
}

.chat-item.active .btn-delete {
  background: white;
  color: #3c5a78;
}

.chat-item.active .btn-delete:hover {
  background: #f7f5f1;
}

.documents-list {
  flex: 1;
  overflow-y: auto;
  padding: 0.5rem;
}

.document-item {
  display: flex;
  align-items: center;
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  border-radius: 0.5rem;
  cursor: pointer;
  transition: background 0.2s;
}

.document-item:hover {
  background: #f7f5f1;
}

.document-item.active {
  background: #3c5a78;
  color: white;
}

.doc-icon {
  font-size: 1.5rem;
  margin-right: 0.75rem;
}

.doc-info {
  flex: 1;
  min-width: 0;
}

.doc-name {
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.doc-meta {
  font-size: 0.75rem;
  opacity: 0.7;
  margin-top: 0.25rem;
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

.message.user .message-text {
  background: #3c5a78;
  color: white;
}

.contexts {
  margin-top: 1rem;
  background: white;
  border: 1px solid #e7e3da;
  border-radius: 0.5rem;
  overflow: hidden;
}

.contexts-header {
  padding: 0.75rem 1rem;
  background: #f7f5f1;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  font-weight: 500;
  font-size: 0.875rem;
}

.context-item {
  padding: 0.75rem 1rem;
  border-top: 1px solid #e7e3da;
  font-size: 0.875rem;
}

.context-score {
  color: #3c5a78;
  font-weight: 500;
  margin-bottom: 0.25rem;
}

.context-text {
  color: #6b7077;
  line-height: 1.5;
}

.understanding {
  margin-top: 0.75rem;
  padding: 1rem;
  background: #fffbeb;
  border-left: 3px solid #f59e0b;
  border-radius: 0.25rem;
  font-size: 0.875rem;
}

.understanding-header {
  font-weight: 600;
  margin-bottom: 0.75rem;
  font-size: 0.9rem;
}

.understanding-step {
  margin-bottom: 0.75rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid #fde68a;
}

.understanding-step:last-child {
  margin-bottom: 0;
  padding-bottom: 0;
  border-bottom: none;
}

.understanding-step strong {
  display: block;
  margin-bottom: 0.25rem;
  color: #92400e;
}

.step-content {
  padding: 0.5rem;
  background: white;
  border-radius: 0.25rem;
  margin-top: 0.25rem;
}

.step-content.highlight {
  background: #fef3c7;
  font-weight: 500;
}

.step-note {
  font-size: 0.8rem;
  color: #78716c;
  margin-top: 0.25rem;
  font-style: italic;
}

.subqueries-list {
  margin: 0.5rem 0 0 0;
  padding-left: 1.5rem;
}

.subqueries-list li {
  margin-bottom: 0.25rem;
  padding: 0.25rem 0;
}

.loading {
  padding: 1rem;
  color: #6b7077;
  font-style: italic;
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

.btn-primary {
  padding: 0.5rem 1rem;
  background: #3c5a78;
  color: white;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
  font-weight: 500;
  width: 100%;
}

.btn-primary:hover:not(:disabled) {
  background: #2e4760;
}

.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-secondary {
  padding: 0.5rem 1rem;
  background: #e7e3da;
  color: #1e2227;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
}

.modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 0.75rem;
  width: 90%;
  max-width: 500px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem;
  border-bottom: 1px solid #e7e3da;
}

.modal-header h3 {
  margin: 0;
}

.btn-close {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0;
  width: 2rem;
  height: 2rem;
}

.modal-body {
  padding: 1.5rem;
}

.upload-area {
  border: 2px dashed #e7e3da;
  border-radius: 0.5rem;
  padding: 3rem;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s;
}

.upload-area:hover {
  border-color: #3c5a78;
}

.upload-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.upload-hint {
  font-size: 0.875rem;
  color: #6b7077;
  margin-top: 0.5rem;
}

.selected-file {
  margin-top: 1rem;
  padding: 1rem;
  background: #f7f5f1;
  border-radius: 0.5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}

.selected-file select {
  padding: 0.5rem;
  border: 1px solid #e7e3da;
  border-radius: 0.25rem;
}

.progress-bar {
  margin-top: 1rem;
  height: 0.5rem;
  background: #e7e3da;
  border-radius: 0.25rem;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #3c5a78;
  transition: width 0.3s;
}

.modal-footer {
  padding: 1.5rem;
  border-top: 1px solid #e7e3da;
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
}

.documents-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
  padding: 1rem 0;
}

.document-card {
  border: 1px solid #e7e3da;
  border-radius: 0.5rem;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  transition: all 0.2s;
}

.document-card:hover {
  border-color: #3c5a78;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.document-card .doc-icon {
  font-size: 2rem;
  text-align: center;
}

.document-card .doc-info {
  flex: 1;
}

.document-card .doc-name {
  font-weight: 500;
  font-size: 0.875rem;
  margin-bottom: 0.25rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.document-card .doc-meta {
  font-size: 0.75rem;
  color: #6b7077;
}

.document-card .doc-date {
  font-size: 0.75rem;
  color: #9ca3af;
  margin-top: 0.25rem;
}

.empty-state {
  text-align: center;
  padding: 2rem 1rem;
  color: #6b7077;
}
</style>

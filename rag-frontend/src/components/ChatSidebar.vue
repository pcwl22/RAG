<template>
  <aside class="sidebar sidebar-left">
    <div class="sidebar-header">
      <h2>💬 对话历史</h2>
      <button @click="emit('new-chat')" class="btn-primary">新建对话</button>
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
        @click="emit('switch-chat', session.id)"
      >
        <div class="chat-icon">💬</div>
        <div class="chat-info">
          <div class="chat-title">{{ session.title }}</div>
          <div class="chat-meta">{{ session.messageCount }} 条消息 · {{ formatTime(session.updatedAt) }}</div>
        </div>
        <button
          v-if="chatSessions.length > 1"
          @click.stop="emit('delete-chat', session.id)"
          class="btn-delete"
          title="删除对话"
        >✕</button>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { formatTime } from '../utils/formatters'

defineProps({
  chatSessions: {
    type: Array,
    default: () => []
  },
  currentSessionId: {
    type: [String, Number, null],
    default: null
  }
})

const emit = defineEmits(['new-chat', 'switch-chat', 'delete-chat'])
</script>

<style scoped>
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

.empty-state {
  text-align: center;
  padding: 2rem 1rem;
  color: #6b7077;
}
</style>

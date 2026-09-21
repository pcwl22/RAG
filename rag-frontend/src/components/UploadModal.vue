<template>
  <div class="modal" @click.self="emit('close')">
    <div class="modal-content">
      <div class="modal-header">
        <h3>📤 上传文档</h3>
        <button @click="emit('close')" class="btn-close">✕</button>
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
          <div class="upload-prompt" @click="fileInput?.click()">
            <div class="upload-icon">📁</div>
            <div>点击选择文件或拖拽到此处</div>
            <div class="upload-hint">支持 PDF, TXT, MD, DOCX, XLSX</div>
          </div>
        </div>

        <div v-if="selectedFile" class="selected-file">
          <div>已选择: {{ selectedFile.name }}</div>
          <select :value="uploadPartition" @change="emit('update:upload-partition', $event.target.value)">
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
        <div v-if="activeTaskId" class="task-id">
          任务 ID：<code>{{ activeTaskId }}</code>
        </div>
      </div>
      <div class="modal-footer">
        <button @click="emit('close')" class="btn-secondary">取消</button>
        <button
          @click="emit('upload')"
          :disabled="!selectedFile || uploading"
          class="btn-primary"
        >
          {{ uploading ? '上传中...' : '上传' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  selectedFile: {
    type: File,
    default: null
  },
  uploadPartition: {
    type: String,
    default: 'general'
  },
  uploadProgress: {
    type: Number,
    default: 0
  },
  uploading: {
    type: Boolean,
    default: false
  },
  activeTaskId: {
    type: String,
    default: ''
  }
})

const emit = defineEmits(['close', 'upload', 'update:selected-file', 'update:upload-partition'])
const fileInput = ref(null)

const handleFileSelect = (event) => {
  emit('update:selected-file', event.target.files[0] || null)
}

const handleDrop = (event) => {
  emit('update:selected-file', event.dataTransfer.files[0] || null)
}
</script>

<style scoped>
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

.task-id {
  margin-top: 0.75rem;
  color: #5f6670;
  font-size: 0.78rem;
  word-break: break-all;
}

.modal-footer {
  padding: 1.5rem;
  border-top: 1px solid #e7e3da;
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
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
</style>

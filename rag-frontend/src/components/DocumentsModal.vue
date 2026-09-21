<template>
  <div class="modal" @click.self="emit('close')">
    <div class="modal-content document-modal-content">
      <div class="modal-header">
        <h3>📚 文档库</h3>
        <button @click="emit('close')" class="btn-close">✕</button>
      </div>
      <div class="modal-body documents-modal-body">
        <div v-if="documentsError" class="documents-error">
          <span>{{ documentsError }}</span>
          <button
            class="btn-secondary btn-small"
            :disabled="documentsLoading"
            @click="emit('change-document-page', documentPage)"
          >重试</button>
        </div>
        <div v-if="documentsLoading" class="documents-loading">正在加载文档...</div>
        <div v-if="!documentsLoading && documents.length === 0" class="empty-state">
          暂无文档
        </div>
        <div v-else class="documents-library">
          <div class="documents-grid">
            <div
              v-for="doc in documents"
              :key="doc.document_id"
              class="document-card"
              :class="{ selected: selectedDoc?.document_id === doc.document_id }"
            >
              <div class="doc-icon">📄</div>
              <div class="doc-info">
                <div class="doc-name" :title="doc.filename">{{ doc.filename }}</div>
                <div class="doc-meta">
                  {{ doc.chunk_count }} 块 · {{ doc.partition }}
                </div>
                <div class="doc-date">{{ formatTime(doc.updated_at) }}</div>
              </div>
              <div class="doc-actions">
                <button @click.stop="emit('load-chunks', doc)" class="btn-secondary btn-small">
                  切片
                </button>
              </div>
            </div>
            <div class="pagination-controls">
              <button
                class="btn-secondary btn-small"
                :disabled="documentPage <= 1 || documentsLoading"
                @click="emit('change-document-page', documentPage - 1)"
              >上一页</button>
              <span>第 {{ documentPage }} / {{ documentPageCount }} 页 · 共 {{ documentTotal }} 个</span>
              <button
                class="btn-secondary btn-small"
                :disabled="documentPage >= documentPageCount || documentsLoading"
                @click="emit('change-document-page', documentPage + 1)"
              >下一页</button>
            </div>
          </div>

          <section class="document-chunks-panel">
            <div v-if="!selectedDoc" class="empty-state">
              选择文件查看真实切片
            </div>
            <template v-else>
              <div class="chunks-panel-header">
                <div>
                  <strong>{{ selectedDoc.filename }}</strong>
                  <span>
                    {{ selectedDocChunks.length }} / {{ selectedDocChunkTotal || selectedDoc.chunk_count || 0 }} 块
                  </span>
                </div>
                <button
                  @click="emit('load-chunks', selectedDoc, selectedDocChunkPage)"
                  class="btn-secondary btn-small"
                >
                  刷新
                </button>
              </div>

              <div v-if="documentChunksLoading" class="empty-state">正在加载切片...</div>
              <div v-else-if="documentChunksError" class="debug-error">{{ documentChunksError }}</div>
              <div v-else-if="selectedDocChunks.length === 0" class="empty-state">暂无切片</div>
              <div v-else class="chunk-list">
                <article
                  v-for="(chunk, idx) in selectedDocChunks"
                  :key="chunk.id || idx"
                  class="chunk-card"
                >
                  <header class="chunk-header" @click="emit('toggle-chunk', chunk)">
                    <div>
                      <strong>#{{ (chunk.chunk_index ?? idx) + 1 }}</strong>
                      <span class="chunk-id">{{ chunk.id }}</span>
                    </div>
                    <span class="toggle">{{ chunk.show ? '▼' : '▶' }}</span>
                  </header>
                  <div class="chunk-meta">
                    <span>长度: {{ chunk.content_length ?? chunk.content?.length ?? 0 }}</span>
                    <span>策略: {{ chunk.chunk_strategy || chunk.metadata?.chunk_strategy || '-' }}</span>
                    <span>分区: {{ chunk.partition || chunk.metadata?.partition || '-' }}</span>
                    <span v-if="chunk.metadata?.level_5_article">条: {{ chunk.metadata.level_5_article }}</span>
                    <span v-if="chunk.metadata?.semantic_chunk_id">语义ID: {{ chunk.metadata.semantic_chunk_id }}</span>
                  </div>
                  <div v-if="hasLegalHierarchy(chunk.metadata)" class="legal-hierarchy compact">
                    <span v-if="chunk.metadata?.level_1_department">部门法: {{ chunk.metadata.level_1_department }}</span>
                    <span v-if="chunk.metadata?.level_3_chapter">章: {{ chunk.metadata.level_3_chapter }}</span>
                    <span v-if="chunk.metadata?.level_4_section">节: {{ chunk.metadata.level_4_section }}</span>
                    <span v-if="chunk.metadata?.level_6_crime_name && chunk.metadata.level_6_crime_name !== '无'">罪名: {{ chunk.metadata.level_6_crime_name }}</span>
                  </div>
                  <div v-show="chunk.show" class="chunk-body">
                    <details open class="context-details">
                      <summary>数据库实际存储 content（真实切片）</summary>
                      <pre class="context-full-text">{{ chunk.content }}</pre>
                    </details>
                    <details v-if="getParentContextText(chunk)" class="context-details">
                      <summary>父块 parent_content</summary>
                      <pre class="context-full-text parent-context-text">{{ getParentContextText(chunk) }}</pre>
                    </details>
                    <details class="context-details">
                      <summary>metadata</summary>
                      <pre class="debug-json">{{ formatJson(chunk.metadata || {}) }}</pre>
                    </details>
                  </div>
                </article>
                <div class="pagination-controls chunk-pagination">
                  <button
                    class="btn-secondary btn-small"
                    :disabled="selectedDocChunkPage <= 1 || documentChunksLoading"
                    @click="emit('change-chunk-page', selectedDoc, selectedDocChunkPage - 1)"
                  >上一页</button>
                  <span>
                    第 {{ selectedDocChunkPage }} / {{ selectedDocChunkPageCount }} 页
                  </span>
                  <button
                    class="btn-secondary btn-small"
                    :disabled="selectedDocChunkPage >= selectedDocChunkPageCount || documentChunksLoading"
                    @click="emit('change-chunk-page', selectedDoc, selectedDocChunkPage + 1)"
                  >下一页</button>
                </div>
              </div>
            </template>
          </section>
        </div>
      </div>
      <div class="modal-footer">
        <button @click="emit('close')" class="btn-secondary">关闭</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { formatJson, formatTime, hasLegalHierarchy } from '../utils/formatters'
import { getParentContextText } from '../utils/contextText'

const props = defineProps({
  documents: {
    type: Array,
    default: () => []
  },
  documentTotal: {
    type: Number,
    default: 0
  },
  documentPage: {
    type: Number,
    default: 1
  },
  documentPageSize: {
    type: Number,
    default: 100
  },
  documentsLoading: {
    type: Boolean,
    default: false
  },
  documentsError: {
    type: String,
    default: ''
  },
  selectedDoc: {
    type: Object,
    default: null
  },
  selectedDocChunks: {
    type: Array,
    default: () => []
  },
  selectedDocChunkTotal: {
    type: Number,
    default: 0
  },
  selectedDocChunkPage: {
    type: Number,
    default: 1
  },
  selectedDocChunkPageSize: {
    type: Number,
    default: 200
  },
  documentChunksLoading: {
    type: Boolean,
    default: false
  },
  documentChunksError: {
    type: String,
    default: ''
  }
})

const documentPageCount = computed(() =>
  Math.max(1, Math.ceil(props.documentTotal / props.documentPageSize))
)
const selectedDocChunkPageCount = computed(() =>
  Math.max(1, Math.ceil(props.selectedDocChunkTotal / props.selectedDocChunkPageSize))
)

const emit = defineEmits([
  'close',
  'load-chunks',
  'toggle-chunk',
  'change-document-page',
  'change-chunk-page'
])
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

.document-modal-content {
  width: min(1180px, 94vw);
  max-width: none;
  max-height: 88vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.document-modal-content .modal-body {
  flex: 1;
  min-height: 0;
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

.documents-modal-body {
  padding: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.documents-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
  padding: 1rem 0;
}

.documents-library {
  flex: 1;
  display: grid;
  grid-template-columns: minmax(240px, 320px) minmax(0, 1fr);
  min-height: 520px;
  height: 100%;
  min-width: 0;
  max-height: calc(88vh - 137px);
  overflow: hidden;
}

.documents-library .documents-grid {
  grid-template-columns: 1fr;
  align-content: start;
  min-height: 0;
  overflow-y: auto;
  padding: 1rem;
  border-right: 1px solid #e7e3da;
}

.documents-loading,
.documents-error {
  padding: 0.65rem 1rem;
  text-align: center;
}

.documents-error {
  color: #991b1b;
  background: #fef2f2;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
}

.pagination-controls {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  padding: 0.6rem 0;
  color: #5f6670;
  font-size: 0.75rem;
}

.pagination-controls button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.chunk-pagination {
  position: sticky;
  bottom: 0;
  background: white;
  border-top: 1px solid #e7e3da;
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

.document-card.selected {
  border-color: #3c5a78;
  background: #f7f9fb;
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

.doc-actions {
  display: flex;
  justify-content: flex-end;
}

.btn-secondary {
  padding: 0.5rem 1rem;
  background: #e7e3da;
  color: #1e2227;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
}

.btn-small {
  width: auto;
  padding: 0.35rem 0.65rem;
  font-size: 0.75rem;
}

.document-chunks-panel {
  min-width: 0;
  min-height: 0;
  padding: 1rem;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.chunks-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid #e7e3da;
}

.chunks-panel-header strong {
  display: block;
  max-width: min(560px, 58vw);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chunks-panel-header span {
  display: block;
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: #6b7077;
}

.chunk-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
  padding-top: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.chunk-card {
  border: 1px solid #e7e3da;
  border-radius: 0.5rem;
  padding: 0.75rem;
  background: #fff;
  flex: 0 0 auto;
}

.chunk-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
  cursor: pointer;
}

.chunk-header > div {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.chunk-id {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #3f4650;
  font-family: Consolas, Monaco, monospace;
  font-size: 0.78rem;
}

.chunk-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.55rem;
  font-size: 0.75rem;
  color: #5f6670;
}

.chunk-meta span {
  padding: 0.18rem 0.45rem;
  border: 1px solid #e7e3da;
  border-radius: 0.35rem;
  background: #f7f5f1;
}

.legal-hierarchy {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin: 0.5rem 0;
  color: #334155;
  font-size: 0.78rem;
}

.legal-hierarchy span {
  padding: 0.25rem 0.4rem;
  background: #eef6ff;
  border: 1px solid #bfdbfe;
  border-radius: 0.25rem;
  word-break: break-word;
}

.legal-hierarchy.compact {
  margin-top: 0.55rem;
}

.chunk-body {
  margin-top: 0.75rem;
}

.context-details {
  margin-top: 0.5rem;
}

.context-details summary {
  cursor: pointer;
  color: #3c5a78;
  font-weight: 600;
  margin-bottom: 0.35rem;
}

.context-full-text {
  margin: 0;
  padding: 0.75rem;
  background: #f8fafc;
  border: 1px solid #e3e8ef;
  border-radius: 0.35rem;
  color: #334155;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 18rem;
  overflow: auto;
  line-height: 1.55;
}

.parent-context-text {
  background: #fff7ed;
  border-color: #fed7aa;
}

.debug-json,
.debug-error {
  margin: 0;
  padding: 0.75rem;
  background: #0f172a;
  color: #e2e8f0;
  border-radius: 0.35rem;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 18rem;
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
  font-size: 0.78rem;
  line-height: 1.45;
}

.debug-error {
  background: #7f1d1d;
  color: #fee2e2;
}

.modal-footer {
  padding: 1.5rem;
  border-top: 1px solid #e7e3da;
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
}

.empty-state {
  text-align: center;
  padding: 2rem 1rem;
  color: #6b7077;
}

@media (max-width: 900px) {
  .documents-library {
    grid-template-columns: 1fr;
    grid-template-rows: auto minmax(0, 1fr);
    min-height: 0;
    max-height: calc(88vh - 137px);
  }

  .documents-library .documents-grid {
    max-height: min(240px, 28vh);
    border-right: none;
    border-bottom: 1px solid #e7e3da;
  }

  .document-chunks-panel {
    min-height: 0;
  }
}
</style>

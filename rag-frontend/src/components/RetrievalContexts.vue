<template>
  <div class="contexts">
    <div class="contexts-header" @click="emit('toggle')">
      📑 检索结果 {{ contexts.length }} 条
      <span class="toggle">{{ show ? '▼' : '▶' }}</span>
    </div>
    <div v-show="show" class="contexts-list">
      <div v-for="(ctx, idx) in contexts" :key="ctx.id || idx" class="context-item">
        <div class="context-score">
          #{{ idx + 1 }} · 得分: {{ formatScore(ctx.score) }}
          <span v-if="ctx.multi_query_appearances" class="query-hit-count">
            命中 {{ ctx.multi_query_appearances }} 次
          </span>
        </div>
        <div class="score-grid">
          <span>ID: {{ ctx.id || '-' }}</span>
          <span>RRF: {{ formatScore(ctx.rrf_score) }}</span>
          <span>向量: {{ formatScore(ctx.vector_score || ctx.metadata?.vector_score) }}</span>
          <span>关键词: {{ formatScore(ctx.bm25_score) }}</span>
          <span>重排: {{ formatScore(ctx.metadata?.rerank_prob) }}</span>
          <span>分区: {{ ctx.partition || ctx.metadata?.partition || '-' }}</span>
        </div>
        <div v-if="ctx.matched_queries && ctx.matched_queries.length" class="matched-queries">
          <div class="matched-title">命中查询</div>
          <div v-for="(matchedQuery, qIdx) in ctx.matched_queries" :key="qIdx" class="matched-query">
            {{ matchedQuery }}
          </div>
        </div>
        <div class="context-location">
          <span>文件: {{ ctx.metadata?.filename || '-' }}</span>
          <span v-if="ctx.metadata?.law_name">法律: {{ ctx.metadata.law_name }}</span>
          <span v-if="ctx.metadata?.legal_citation">定位: {{ ctx.metadata.legal_citation }}</span>
          <span v-if="ctx.metadata?.chunk_index !== undefined">块: {{ ctx.metadata.chunk_index }}</span>
        </div>
        <div v-if="hasLegalHierarchy(ctx.metadata)" class="legal-hierarchy">
          <span v-if="ctx.metadata?.level_1_department">部门法: {{ ctx.metadata.level_1_department }}</span>
          <span v-if="ctx.metadata?.level_2_part">编/部分: {{ ctx.metadata.level_2_part }}</span>
          <span v-if="ctx.metadata?.level_3_chapter">章: {{ ctx.metadata.level_3_chapter }}</span>
          <span v-if="ctx.metadata?.level_4_section">节: {{ ctx.metadata.level_4_section }}</span>
          <span v-if="ctx.metadata?.level_5_article">条: {{ ctx.metadata.level_5_article }}</span>
          <span v-if="ctx.metadata?.level_6_crime_name && ctx.metadata.level_6_crime_name !== '无'">罪名: {{ ctx.metadata.level_6_crime_name }}</span>
          <span v-if="ctx.metadata?.parent_law_id">父法: {{ ctx.metadata.parent_law_id }}</span>
          <span v-if="ctx.metadata?.semantic_chunk_id">语义ID: {{ ctx.metadata.semantic_chunk_id }}</span>
          <span v-if="formatMetadataList(ctx.metadata?.keywords)">关键词: {{ formatMetadataList(ctx.metadata.keywords) }}</span>
        </div>
        <details open class="context-details">
          <summary>实际送入 LLM 的文本（child_content / article_text）</summary>
          <pre class="context-full-text">{{ getModelContextText(ctx) }}</pre>
        </details>
        <details v-if="getParentContextText(ctx)" class="context-details">
          <summary>父块 parent_content（仅供定位和扩展参考）</summary>
          <pre class="context-full-text parent-context-text">{{ getParentContextText(ctx) }}</pre>
        </details>
        <details v-if="getStoredContextText(ctx)" class="context-details">
          <summary>检索返回 content 字段</summary>
          <pre class="context-full-text">{{ getStoredContextText(ctx) }}</pre>
        </details>
        <details class="context-details">
          <summary>metadata</summary>
          <pre class="debug-json">{{ formatJson(ctx.metadata || {}) }}</pre>
        </details>
      </div>
    </div>
  </div>
</template>

<script setup>
import {
  formatJson,
  formatMetadataList,
  formatScore,
  hasLegalHierarchy
} from '../utils/formatters'
import {
  getModelContextText,
  getParentContextText,
  getStoredContextText
} from '../utils/contextText'

defineProps({
  contexts: {
    type: Array,
    default: () => []
  },
  show: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['toggle'])
</script>

<style scoped>
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

.query-hit-count {
  margin-left: 0.5rem;
  padding: 0.1rem 0.4rem;
  border-radius: 0.25rem;
  background: #e7e3da;
  color: #4b5563;
  font-size: 0.75rem;
  font-weight: 500;
}

.score-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.35rem;
  margin: 0.5rem 0;
  color: #475569;
  font-size: 0.78rem;
}

.score-grid span,
.context-location span,
.legal-hierarchy span {
  padding: 0.25rem 0.4rem;
  background: #f8fafc;
  border: 1px solid #e3e8ef;
  border-radius: 0.25rem;
  word-break: break-word;
}

.matched-queries {
  margin: 0.35rem 0 0.5rem;
  padding: 0.5rem;
  background: #f7f5f1;
  border-radius: 0.25rem;
  color: #4b5563;
}

.matched-title {
  margin-bottom: 0.25rem;
  color: #3c5a78;
  font-size: 0.75rem;
  font-weight: 600;
}

.matched-query {
  line-height: 1.4;
  word-break: break-word;
}

.context-location {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin: 0.5rem 0;
  color: #475569;
  font-size: 0.78rem;
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
  background: #eef6ff;
  border-color: #bfdbfe;
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

.debug-json {
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
</style>

<template>
  <div class="understanding">
    <div class="understanding-header" @click="emit('toggle')">
      <span>🧠 查询理解过程</span>
      <span class="toggle">{{ show ? '▼' : '▶' }}</span>
    </div>

    <div v-show="show" class="understanding-body">
      <div class="understanding-step">
        <strong>原始查询：</strong>
        <div class="step-content">{{ understanding.original_query }}</div>
      </div>

      <div v-if="understanding.resolved_query !== understanding.original_query" class="understanding-step">
        <strong>✓ 指代消解：</strong>
        <div class="step-content highlight">{{ understanding.resolved_query }}</div>
        <div class="step-note">将代词还原为明确实体</div>
      </div>

      <div v-if="understanding.rewritten_query && understanding.rewritten_query !== understanding.resolved_query" class="understanding-step">
        <strong>✓ 查询改写：</strong>
        <div class="step-content highlight">{{ understanding.rewritten_query }}</div>
        <div class="step-note">用于和原查询一起检索，提升召回</div>
      </div>

      <div v-if="understanding.subqueries && understanding.subqueries.length > 1" class="understanding-step">
        <strong>✓ 查询拆分：</strong>
        <div class="step-note">复合问题拆分为 {{ understanding.subqueries.length }} 个子查询</div>
        <ol class="subqueries-list">
          <li v-for="(sq, i) in understanding.subqueries" :key="i">{{ sq }}</li>
        </ol>
      </div>

      <div v-if="understanding.retrieval_queries && understanding.retrieval_queries.length > 0" class="understanding-step">
        <strong>✓ 实际检索查询：</strong>
        <ol class="subqueries-list">
          <li v-for="(rq, i) in understanding.retrieval_queries" :key="i">{{ rq }}</li>
        </ol>
      </div>

      <div v-if="understanding.concept_article_mappings && understanding.concept_article_mappings.length > 0" class="understanding-step">
        <strong>✓ 法律概念-法条映射：</strong>
        <div class="concept-map-list">
          <div
            v-for="mapping in understanding.concept_article_mappings"
            :key="mapping.concept_id"
            class="concept-map-item"
          >
            <div class="concept-map-title">
              {{ mapping.concept }}
              <span v-if="mapping.matched_terms?.length">命中: {{ mapping.matched_terms.join('、') }}</span>
            </div>
            <div class="concept-articles">
              <span
                v-for="article in mapping.articles"
                :key="`${mapping.concept_id}-${article.semantic_chunk_id || article.article}`"
              >
                {{ article.law }} {{ article.article }}
              </span>
            </div>
            <div class="step-note">{{ mapping.note }}</div>
          </div>
        </div>
      </div>

      <div v-if="hasSignalEntries(understanding.retrieval_signals)" class="understanding-step">
        <strong>✓ 检索语义信号：</strong>
        <div class="signal-grid">
          <div v-for="[key, values] in signalEntries(understanding.retrieval_signals)" :key="key" class="signal-item">
            <span>{{ key }}</span>
            <strong>{{ values.join('、') }}</strong>
          </div>
        </div>
      </div>

      <div
        v-if="!understanding.is_decomposed && understanding.resolved_query === understanding.original_query && understanding.rewritten_query === understanding.resolved_query"
        class="understanding-step"
      >
        <strong>✓ 分析结果：</strong>
        <div class="step-content">单一问题，无需拆分或改写</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { hasSignalEntries, signalEntries } from '../utils/formatters'

defineProps({
  understanding: {
    type: Object,
    required: true
  },
  show: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['toggle'])
</script>

<style scoped>
.understanding {
  margin-top: 0.75rem;
  padding: 1rem;
  background: #fffbeb;
  border-left: 3px solid #f59e0b;
  border-radius: 0.25rem;
  font-size: 0.875rem;
}

.understanding-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  user-select: none;
}

.understanding-body {
  margin-top: 0.75rem;
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

.concept-map-list {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.concept-map-item {
  padding: 0.65rem;
  background: white;
  border: 1px solid #fde68a;
  border-radius: 0.35rem;
}

.concept-map-title {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  font-weight: 600;
  color: #92400e;
}

.concept-map-title span {
  font-weight: 400;
  color: #6b7077;
}

.concept-articles {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin-top: 0.45rem;
}

.concept-articles span {
  padding: 0.18rem 0.45rem;
  border-radius: 0.35rem;
  background: #fef3c7;
  color: #78350f;
  font-size: 0.75rem;
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

.signal-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.45rem;
  margin-top: 0.5rem;
}

.signal-item {
  padding: 0.5rem;
  background: white;
  border-radius: 0.25rem;
}

.signal-item span {
  display: block;
  color: #92400e;
  font-size: 0.75rem;
  margin-bottom: 0.2rem;
}

.signal-item strong {
  color: #1e2227;
  font-weight: 500;
}
</style>

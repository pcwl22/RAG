<template>
  <div class="process-trace">
    <div class="process-header" @click="emit('toggle')">
      <div>
        <strong>RAG 过程追踪</strong>
        <span class="process-subtitle">
          {{ process.mode }} · {{ process.endpoint }}
        </span>
      </div>
      <div class="process-header-right">
        <span v-if="process.streaming" class="process-badge running">进行中</span>
        <span v-else-if="process.error" class="process-badge error">异常</span>
        <span v-else class="process-badge done">完成</span>
        <span class="toggle">{{ process.show ? '▼' : '▶' }}</span>
      </div>
    </div>

    <div v-show="process.show" class="process-body">
      <div class="process-summary">
        <div>
          <span>开始时间</span>
          <strong>{{ process.startedAt }}</strong>
        </div>
        <div>
          <span>检索结果</span>
          <strong>{{ contextCount }} 条</strong>
        </div>
        <div>
          <span>流式片段</span>
          <strong>{{ process.chunkCount || 0 }} 段 / {{ process.chunkChars || 0 }} 字</strong>
        </div>
        <div>
          <span>耗时</span>
          <strong>{{ formatDuration(process.totalTime) }}</strong>
        </div>
      </div>

      <div v-if="process.steps && process.steps.length" class="process-section">
        <div class="process-section-title">状态时间线</div>
        <div class="timeline">
          <div v-for="(step, stepIndex) in process.steps" :key="stepIndex" class="timeline-item">
            <div class="timeline-time">{{ step.time }}</div>
            <div class="timeline-content">
              <strong>{{ step.label }}</strong>
              <span v-if="step.detail">{{ step.detail }}</span>
            </div>
          </div>
        </div>
      </div>

      <div class="process-section">
        <div class="process-section-title">请求参数</div>
        <pre class="debug-json">{{ formatJson(process.request) }}</pre>
      </div>

      <div v-if="process.events && process.events.length" class="process-section">
        <div class="process-section-title">SSE / 响应事件</div>
        <div class="event-list">
          <div v-for="(event, eventIndex) in process.events" :key="eventIndex" class="event-item">
            <div class="event-meta">
              <span>{{ event.time }}</span>
              <strong>{{ event.type }}</strong>
            </div>
            <pre class="event-payload">{{ formatJson(event.data) }}</pre>
          </div>
        </div>
      </div>

      <div v-if="process.subAnswers && process.subAnswers.length" class="process-section">
        <div class="process-section-title">子问题答案</div>
        <div v-for="(item, subIndex) in process.subAnswers" :key="subIndex" class="sub-answer">
          <strong>{{ subIndex + 1 }}. {{ item.query }}</strong>
          <div class="sub-answer-meta">命中文档 {{ item.result_count ?? '-' }} 条</div>
          <pre>{{ item.answer }}</pre>
        </div>
      </div>

      <div v-if="process.error" class="process-section">
        <div class="process-section-title">错误信息</div>
        <pre class="debug-error">{{ process.error }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup>
import { formatDuration, formatJson } from '../utils/formatters'

defineProps({
  process: {
    type: Object,
    required: true
  },
  contextCount: {
    type: Number,
    default: 0
  }
})

const emit = defineEmits(['toggle'])
</script>

<style scoped>
.process-trace {
  margin-top: 0.75rem;
  border: 1px solid #d6dde6;
  border-radius: 0.5rem;
  overflow: hidden;
  background: #fbfcfd;
  font-size: 0.875rem;
}

.process-header {
  padding: 0.75rem 1rem;
  background: #edf2f7;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
}

.process-subtitle {
  display: block;
  margin-top: 0.2rem;
  color: #596579;
  font-size: 0.78rem;
  word-break: break-word;
}

.process-header-right {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-shrink: 0;
}

.process-badge {
  padding: 0.15rem 0.45rem;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  font-weight: 600;
}

.process-badge.running {
  background: #dbeafe;
  color: #1d4ed8;
}

.process-badge.done {
  background: #d1fae5;
  color: #047857;
}

.process-badge.error {
  background: #fee2e2;
  color: #b91c1c;
}

.process-body {
  padding: 1rem;
}

.process-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.process-summary div {
  padding: 0.6rem;
  background: white;
  border: 1px solid #e3e8ef;
  border-radius: 0.35rem;
}

.process-summary span {
  display: block;
  color: #64748b;
  font-size: 0.75rem;
  margin-bottom: 0.25rem;
}

.process-summary strong {
  color: #1e293b;
  word-break: break-word;
}

.process-section {
  margin-top: 1rem;
}

.process-section-title {
  margin-bottom: 0.45rem;
  color: #334155;
  font-weight: 700;
}

.timeline {
  border-left: 2px solid #cbd5e1;
  margin-left: 0.35rem;
}

.timeline-item {
  display: grid;
  grid-template-columns: 5.8rem 1fr;
  gap: 0.75rem;
  padding: 0.35rem 0 0.45rem 0.75rem;
  position: relative;
}

.timeline-item::before {
  content: '';
  position: absolute;
  left: -0.35rem;
  top: 0.75rem;
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 50%;
  background: #3c5a78;
}

.timeline-time {
  color: #64748b;
  font-variant-numeric: tabular-nums;
}

.timeline-content strong {
  display: block;
  color: #1e293b;
}

.timeline-content span {
  color: #475569;
  word-break: break-word;
}

.debug-json,
.event-payload,
.debug-error,
.sub-answer pre {
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

.event-list {
  display: grid;
  gap: 0.5rem;
}

.event-item {
  border: 1px solid #e3e8ef;
  border-radius: 0.35rem;
  background: white;
  overflow: hidden;
}

.event-meta {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.45rem 0.65rem;
  background: #f8fafc;
  color: #475569;
}

.event-meta strong {
  color: #1e293b;
}

.event-payload {
  border-radius: 0;
  max-height: 12rem;
}

.sub-answer {
  margin-bottom: 0.75rem;
  padding: 0.75rem;
  border: 1px solid #e3e8ef;
  border-radius: 0.35rem;
  background: white;
}

.sub-answer strong {
  display: block;
  margin-bottom: 0.25rem;
  color: #1e293b;
}

.sub-answer-meta {
  color: #64748b;
  font-size: 0.78rem;
  margin-bottom: 0.5rem;
}
</style>

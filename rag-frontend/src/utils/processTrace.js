import { API_BASE } from '../api/ragApi'
import { formatDuration, truncateText } from './formatters'

const nowLabel = () => new Date().toLocaleTimeString()

const safeClone = (value) => {
  try {
    return JSON.parse(JSON.stringify(value))
  } catch (error) {
    return String(value)
  }
}

const compactDebugValue = (value) => {
  if (value === null || value === undefined) return value
  if (typeof value === 'string') return truncateText(value, 600)
  if (typeof value !== 'object') return value
  if (Array.isArray(value)) {
    const items = value.slice(0, 8).map(compactDebugValue)
    if (value.length > 8) items.push(`... 另有 ${value.length - 8} 项`)
    return items
  }

  const entries = Object.entries(value)
  const compacted = {}
  for (const [key, item] of entries.slice(0, 24)) {
    compacted[key] = compactDebugValue(item)
  }
  if (entries.length > 24) compacted.__truncated__ = `另有 ${entries.length - 24} 个字段`
  return compacted
}

const sanitizeRequestPayload = (payload = {}) => {
  const cloned = safeClone(payload)
  const sanitize = (value) => {
    if (!value || typeof value !== 'object') return value
    for (const key of Object.keys(value)) {
      if (/api[_-]?key|token|secret|password/i.test(key)) {
        value[key] = '***'
      } else {
        value[key] = sanitize(value[key])
      }
    }
    return value
  }
  return sanitize(cloned)
}

export const createProcessTrace = ({ mode, endpoint, request }) => ({
  mode,
  endpoint: endpoint.replace(API_BASE, ''),
  request: sanitizeRequestPayload(request),
  startedAt: new Date().toLocaleString(),
  totalTime: null,
  streaming: false,
  show: false,
  steps: [],
  events: [],
  subAnswers: [],
  chunkCount: 0,
  chunkChars: 0,
  lastChunkPreview: '',
  error: ''
})

export const addProcessStep = (process, label, detail = '') => {
  if (!process) return
  process.steps.push({
    time: nowLabel(),
    label,
    detail
  })
}

export const addProcessEvent = (process, payload) => {
  if (!process || !payload) return
  const type = payload.type || 'response'

  if (type === 'chunk' || type === 'token') {
    const chunk = payload.data || ''
    process.chunkCount += 1
    process.chunkChars += chunk.length
    process.lastChunkPreview = truncateText(chunk, 160)
    if (process.events.length < 120) {
      process.events.push({
        time: nowLabel(),
        type,
        data: {
          length: chunk.length,
          preview: truncateText(chunk, 220)
        }
      })
    }
    return
  }

  let data = payload.data ?? payload.error ?? payload
  if (type === 'sources' && Array.isArray(payload.data)) {
    data = {
      count: payload.data.length,
      top: payload.data.slice(0, 5).map((doc, index) => ({
        rank: index + 1,
        id: doc.id,
        score: doc.score,
        rrf_score: doc.rrf_score,
        filename: doc.metadata?.filename,
        citation: doc.metadata?.legal_citation,
        matched_queries: doc.matched_queries
      }))
    }
  }

  process.events.push({
    time: nowLabel(),
    type,
    data: compactDebugValue(data)
  })
}

export const finishProcess = (process, totalTime = null) => {
  if (!process) return
  process.streaming = false
  process.totalTime = totalTime
  addProcessStep(process, '流程结束', totalTime === null ? '' : `后端耗时 ${formatDuration(totalTime)}`)
}

export const normalizeProcessTrace = (process) => {
  if (!process) return process
  return {
    ...process,
    streaming: false,
    show: false,
    steps: process.steps || [],
    events: process.events || [],
    subAnswers: process.subAnswers || [],
    chunkCount: process.chunkCount || 0,
    chunkChars: process.chunkChars || 0,
    error: process.error || ''
  }
}

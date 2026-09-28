export const truncateText = (value = '', maxLength = 180) => {
  const text = String(value)
  return text.length > maxLength ? text.slice(0, maxLength) + '...' : text
}

const TIME_WITHOUT_ZONE = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?$/

export const parseTimestamp = (value) => {
  if (!value) return null
  const text = String(value).trim()
  if (!text) return null
  const normalized = TIME_WITHOUT_ZONE.test(text)
    ? `${text.replace(' ', 'T')}Z`
    : text
  const date = new Date(normalized)
  return Number.isNaN(date.getTime()) ? null : date
}

export const formatTime = (dateStr) => {
  const date = parseTimestamp(dateStr)
  if (!date) return ''
  const now = new Date()
  const diff = now - date

  if (diff < 60000) return '刚刚'
  if (diff < 3600000) return Math.floor(diff / 60000) + '分钟前'
  if (diff < 86400000) return Math.floor(diff / 3600000) + '小时前'
  if (diff < 604800000) return Math.floor(diff / 86400000) + '天前'

  return date.toLocaleDateString()
}

export const formatDuration = (seconds) => {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return '-'
  const value = Number(seconds)
  if (value < 1) return `${Math.round(value * 1000)}ms`
  return `${value.toFixed(2)}s`
}

export const formatScore = (score) => {
  if (score === null || score === undefined || Number.isNaN(Number(score))) return '-'
  return Number(score).toFixed(4)
}

export const formatJson = (value) => {
  try {
    return JSON.stringify(value ?? {}, null, 2)
  } catch (error) {
    return String(value)
  }
}

export const formatMetadataList = (value) => {
  if (Array.isArray(value)) return value.filter(Boolean).join('、')
  if (typeof value === 'string') return value
  return ''
}

export const hasLegalHierarchy = (metadata = {}) =>
  Boolean(
    metadata.level_1_department ||
    metadata.level_2_part ||
    metadata.level_3_chapter ||
    metadata.level_4_section ||
    metadata.level_5_article ||
    metadata.level_6_crime_name ||
    metadata.parent_law_id ||
    metadata.semantic_chunk_id ||
    formatMetadataList(metadata.keywords)
  )

export const signalEntries = (signals = {}) =>
  Object.entries(signals)
    .filter(([, values]) => Array.isArray(values) && values.length > 0)

export const hasSignalEntries = (signals = {}) => signalEntries(signals).length > 0

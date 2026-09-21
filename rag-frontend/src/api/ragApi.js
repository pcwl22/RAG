import { authorizedFetch } from '../auth/oidc.js'

// Default to same-origin so the frontend and API share one public origin.
export const API_BASE = import.meta.env?.VITE_API_BASE || '/api/v1'
export const API_ROOT = API_BASE.replace(/\/api\/v1\/?$/, '')

export const ENDPOINTS = {
  answer: `${API_BASE}/answer`,
  enhancedQuery: `${API_BASE}/query/enhanced`
}

export class ApiError extends Error {
  constructor(message, { status = 0, retryAfterSeconds = null } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.retryAfterSeconds = retryAfterSeconds
  }

  get retryable() {
    return this.status === 0 || [408, 425, 429, 500, 502, 503, 504].includes(this.status)
  }
}

export const ensureOk = async (response) => {
  if (response.ok) return response

  let detail = ''
  try {
    const data = await response.json()
    detail = data.detail || data.message || data.error || ''
  } catch (error) {
    detail = ''
  }

  const retryAfterHeader = response.headers?.get?.('Retry-After')
  const retryAfter = retryAfterHeader === null || retryAfterHeader === undefined
    ? Number.NaN
    : Number(retryAfterHeader)
  throw new ApiError(
    detail ? `HTTP ${response.status}: ${detail}` : `HTTP ${response.status}`,
    {
      status: response.status,
      retryAfterSeconds: Number.isFinite(retryAfter) && retryAfter >= 0 ? retryAfter : null
    }
  )
}

export const fetchHealth = async () => {
  try {
    const response = await fetch(`${API_ROOT}/health/ready`)
    if (!response.ok) return false
    const data = await response.json()
    return data.status === 'ready'
  } catch (error) {
    return false
  }
}

export const fetchDocuments = async ({ skip = 0, limit = 100, signal } = {}) => {
  const query = new URLSearchParams({ skip: String(skip), limit: String(limit) })
  const response = await ensureOk(
    await authorizedFetch(`${API_BASE}/documents?${query}`, { signal })
  )
  return response.json()
}

export const fetchDocumentChunks = async (
  documentId,
  { skip = 0, limit = 200, signal } = {}
) => {
  const encodedId = encodeURIComponent(documentId)
  const query = new URLSearchParams({ skip: String(skip), limit: String(limit) })
  const response = await ensureOk(
    await authorizedFetch(`${API_BASE}/documents/${encodedId}/chunks?${query}`, { signal })
  )
  return response.json()
}

export const postJson = async (endpoint, payload) => {
  const response = await ensureOk(
    await authorizedFetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
  )
  return response.json()
}

export const postStream = async (endpoint, payload, { signal } = {}) =>
  ensureOk(
    await authorizedFetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal
    })
  )

export const ingestDocument = async (file, partition) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('partition', partition)

  const response = await ensureOk(
    await authorizedFetch(`${API_BASE}/documents/ingest`, {
      method: 'POST',
      body: formData
    })
  )
  return response.json()
}

export const fetchDocumentStatus = async (taskId) => {
  const encodedTaskId = encodeURIComponent(taskId)
  const response = await ensureOk(
    await authorizedFetch(`${API_BASE}/documents/status/${encodedTaskId}`)
  )
  return response.json()
}

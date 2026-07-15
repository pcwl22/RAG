// Default to same-origin. The reverse proxy/BFF owns upstream routing and any
// server-side API credential; permanent secrets must not be bundled by Vite.
export const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1'
export const API_ROOT = API_BASE.replace(/\/api\/v1\/?$/, '')

export const ENDPOINTS = {
  answer: `${API_BASE}/answer`,
  enhancedQuery: `${API_BASE}/query/enhanced`
}

const ensureOk = async (response) => {
  if (response.ok) return response

  let detail = ''
  try {
    const data = await response.json()
    detail = data.detail || data.message || data.error || ''
  } catch (error) {
    detail = ''
  }

  throw new Error(detail ? `HTTP ${response.status}: ${detail}` : `HTTP ${response.status}`)
}

export const fetchHealth = async () => {
  try {
    const response = await fetch(`${API_ROOT}/health`)
    return response.ok
  } catch (error) {
    return false
  }
}

export const fetchDocuments = async () => {
  const response = await ensureOk(await fetch(`${API_BASE}/documents`))
  return response.json()
}

export const fetchDocumentChunks = async (documentId, { limit = 5000 } = {}) => {
  const encodedId = encodeURIComponent(documentId)
  const response = await ensureOk(
    await fetch(`${API_BASE}/documents/${encodedId}/chunks?limit=${limit}`)
  )
  return response.json()
}

export const postJson = async (endpoint, payload) => {
  const response = await ensureOk(
    await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
  )
  return response.json()
}

export const postStream = async (endpoint, payload) =>
  ensureOk(
    await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
  )

export const ingestDocument = async (file, partition) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('partition', partition)

  return ensureOk(
    await fetch(`${API_BASE}/documents/ingest`, {
      method: 'POST',
      body: formData
    })
  )
}

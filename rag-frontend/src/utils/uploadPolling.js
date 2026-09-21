const DEFAULT_RETRYABLE_STATUSES = new Set([0, 408, 425, 429, 500, 502, 503, 504])
const SAFE_TASK_ID = /^[A-Za-z0-9_-]{1,128}$/

export const isRetryablePollingError = (error) =>
  error?.retryable === true || DEFAULT_RETRYABLE_STATUSES.has(Number(error?.status || 0))

export const pollDocumentTask = async (
  taskId,
  {
    fetchStatus,
    delay,
    onProgress = () => {},
    maxPolls = 300,
    baseDelayMs = 1000,
    maxDelayMs = 10000
  }
) => {
  if (!SAFE_TASK_ID.test(String(taskId || ''))) {
    const invalid = new Error('上传任务 ID 格式无效')
    invalid.terminal = true
    throw invalid
  }
  let transientFailures = 0
  for (let attempt = 0; attempt < maxPolls; attempt += 1) {
    try {
      const status = await fetchStatus(taskId)
      transientFailures = 0
      onProgress(status)
      if (status.status === 'completed') return status
      if (status.status === 'failed') {
        const failure = new Error(status.error || '文档处理失败')
        failure.terminal = true
        throw failure
      }
      await delay(baseDelayMs)
    } catch (error) {
      if (error?.terminal === true) throw error
      if (!isRetryablePollingError(error)) throw error
      transientFailures += 1
      const hasServerDelay = error?.retryAfterSeconds !== null &&
        error?.retryAfterSeconds !== undefined &&
        Number.isFinite(Number(error.retryAfterSeconds))
      const serverDelay = Number(error?.retryAfterSeconds) * 1000
      const backoff = Math.min(maxDelayMs, baseDelayMs * (2 ** Math.min(transientFailures, 4)))
      await delay(hasServerDelay && serverDelay >= 0 ? serverDelay : backoff)
    }
  }
  throw new Error(`文档任务 ${taskId} 处理超时，请保留任务 ID 后重试`)
}

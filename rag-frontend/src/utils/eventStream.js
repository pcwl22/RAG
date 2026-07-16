export const createStreamWriter = (write) => {
  let queue = ''
  let timer = null
  let drainResolvers = []

  const resolveDrains = () => {
    const resolvers = drainResolvers
    drainResolvers = []
    resolvers.forEach(resolve => resolve())
  }

  const stop = () => {
    if (timer) clearInterval(timer)
    timer = null
  }

  const flush = () => {
    if (!queue) {
      stop()
      resolveDrains()
      return
    }

    const size = queue.length > 160 ? 16 : queue.length > 60 ? 8 : 3
    const next = queue.slice(0, size)
    queue = queue.slice(size)
    write(next)
  }

  const start = () => {
    if (!timer) timer = setInterval(flush, 18)
  }

  return {
    enqueue(text = '') {
      if (!text) return
      queue += text
      start()
    },
    drain() {
      if (!queue && !timer) return Promise.resolve()
      start()
      return new Promise(resolve => {
        drainResolvers.push(resolve)
      })
    }
  }
}

export const readEventStream = async (res, handlers = {}) => {
  if (!res.body) throw new Error('浏览器不支持流式响应')

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let finished = false

  const handleEvent = (event) => {
    const rawData = event
      .split('\n')
      .filter(line => line.startsWith('data:'))
      .map(line => line.slice(5).trimStart())
      .join('\n')
      .trim()

    if (!rawData) return
    if (rawData === '[DONE]') {
      finished = true
      handlers.onDone?.()
      return
    }

    let payload
    try {
      payload = JSON.parse(rawData)
    } catch (error) {
      throw new Error(`流式响应格式错误: ${rawData.slice(0, 120)}`)
    }

    if (payload.type === 'error' || payload.error) {
      handlers.onEvent?.(payload)
      throw new Error(payload.error || payload.message || '流式响应失败')
    }

    handlers.onEvent?.(payload)
    if (payload.type === 'chunk' || payload.type === 'token') handlers.onChunk?.(payload.data || '')
    if (payload.type === 'sources') handlers.onSources?.(payload.data)
    if (payload.type === 'understanding') handlers.onUnderstanding?.(payload.data)
    if (payload.type === 'status') handlers.onStatus?.(payload.data || '')
    if (payload.type === 'done') {
      finished = true
      handlers.onDone?.(payload)
    }
  }

  try {
    while (!finished) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      buffer = buffer.replace(/\r\n/g, '\n')
      const events = buffer.split('\n\n')
      buffer = events.pop() || ''

      for (const event of events) {
        handleEvent(event)
        if (finished) break
      }
    }

    buffer += decoder.decode()
    if (!finished && buffer.trim()) handleEvent(buffer)
    if (!finished) throw new Error('流式响应意外中断')
  } finally {
    reader.releaseLock()
  }
}

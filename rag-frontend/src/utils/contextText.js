export const getModelContextText = (ctx = {}) => {
  const metadata = ctx.metadata || {}
  return String(
    ctx.child_content ||
    metadata.child_content ||
    metadata.article_text ||
    ctx.content ||
    ''
  )
}

export const getParentContextText = (ctx = {}) => {
  const metadata = ctx.metadata || {}
  const parent = metadata.parent_content || ctx.parent_content || ''
  const modelText = getModelContextText(ctx)
  if (!parent || parent === modelText) return ''
  return String(parent)
}

export const getStoredContextText = (ctx = {}) => {
  const content = ctx.content || ''
  const modelText = getModelContextText(ctx)
  const parentText = getParentContextText(ctx)
  if (!content || content === modelText || content === parentText) return ''
  return String(content)
}

export const getRelevantSnippet = (content, query) => {
  if (!content || !query) return content?.substring(0, 200) + '...'

  const keywords = query.replace(/[？?！!，,。.]/g, ' ').split(' ').filter(w => w.length > 1)

  let matchIndex = -1
  for (const keyword of keywords) {
    matchIndex = content.indexOf(keyword)
    if (matchIndex !== -1) break
  }

  if (matchIndex !== -1) {
    const start = Math.max(0, matchIndex - 50)
    const end = Math.min(content.length, matchIndex + 300)
    let snippet = content.substring(start, end)

    if (start > 0) snippet = '...' + snippet
    if (end < content.length) snippet = snippet + '...'

    return snippet
  }

  return content.substring(0, 200) + (content.length > 200 ? '...' : '')
}

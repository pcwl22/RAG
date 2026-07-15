import test from 'node:test'
import assert from 'node:assert/strict'

import { getModelContextText, getParentContextText, getRelevantSnippet } from '../src/utils/contextText.js'
import { formatDuration, formatScore, truncateText } from '../src/utils/formatters.js'

test('context helpers preserve model-visible child content', () => {
  const context = {
    content: 'stored child',
    child_content: 'model child',
    metadata: { parent_content: 'parent text' }
  }
  assert.equal(getModelContextText(context), 'model child')
  assert.equal(getParentContextText(context), 'parent text')
  assert.match(getRelevantSnippet('前文劳动合同解除后文', '劳动合同'), /劳动合同/)
})

test('formatters handle boundaries', () => {
  assert.equal(truncateText('abcdef', 3), 'abc...')
  assert.equal(formatDuration(0.25), '250ms')
  assert.equal(formatScore(0.12345), '0.1235')
})

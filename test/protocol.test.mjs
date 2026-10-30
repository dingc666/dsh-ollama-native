import assert from 'node:assert/strict'
import { test } from 'node:test'

import { buildChatBody } from '../lib/body.js'
import { decodeOllamaStream } from '../lib/chunks.js'
import { splitNdjson } from '../lib/ndjson.js'
import { wireMessages } from '../lib/wire.js'

/** Build a byte stream from string pieces, to exercise partial framing. */
function byteStream(pieces) {
  return new ReadableStream({
    start(controller) {
      for (const piece of pieces) controller.enqueue(new TextEncoder().encode(piece))
      controller.close()
    },
  })
}

async function lines(pieces) {
  const collected = []
  for await (const line of splitNdjson(byteStream(pieces))) collected.push(line)
  return collected
}

async function chunks(pieces, options) {
  const collected = []
  for await (const chunk of decodeOllamaStream(splitNdjson(byteStream(pieces)), options)) collected.push(chunk)
  return collected
}

const line = (value) => `${JSON.stringify(value)}\n`

test('frames NDJSON across arbitrary chunk boundaries', async () => {
  const received = await lines(['{"a":1}\r\n{"b"', ':2}\n\n{"c":3}'])
  assert.deepEqual(received, ['{"a":1}', '{"b":2}', '{"c":3}'])
})

test('translates text, usage and a stop', async () => {
  const received = await chunks([
    line({ message: { role: 'assistant', content: 'Hel' } }),
    line({ message: { content: 'lo' }, done: true, done_reason: 'stop', prompt_eval_count: 5, eval_count: 2 }),
  ])

  assert.deepEqual(received, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'Hel' },
    { type: 'text-delta', index: 0, text: 'lo' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Hello' } },
    { type: 'usage', usage: { inputTokens: 5, outputTokens: 2, totalTokens: 7 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
})

test('puts a reasoning block before the answer', async () => {
  const received = await chunks([
    line({ message: { thinking: 'think ' } }),
    line({ message: { thinking: 'harder' } }),
    line({ message: { content: 'ok' }, done: true, done_reason: 'stop' }),
  ])

  assert.deepEqual(
    received.map((chunk) => `${chunk.type}:${chunk.index}`),
    ['block-start:0', 'reasoning-delta:0', 'reasoning-delta:0', 'block-start:1', 'text-delta:1', 'block-end:0', 'block-end:1', 'finish:undefined'],
  )
  assert.deepEqual(received[5], { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'think harder' } })
})

test('emits a tool call and finishes with tool-calls even when the server says stop', async () => {
  const received = await chunks([
    line({
      message: {
        role: 'assistant',
        content: '',
        tool_calls: [{ id: 'call_1', function: { name: 'read_file', arguments: { path: 'a.txt' } } }],
      },
      done: true,
      done_reason: 'stop',
    }),
  ])

  const call = received.find((chunk) => chunk.type === 'tool-call-delta')
  assert.equal(call.id, 'call_1')
  assert.equal(call.name, 'read_file')
  assert.deepEqual(JSON.parse(call.argumentsDelta), { path: 'a.txt' })

  const end = received.find((chunk) => chunk.type === 'block-end')
  assert.deepEqual(end.block, {
    type: 'tool-call',
    id: 'call_1',
    name: 'read_file',
    arguments: JSON.stringify({ path: 'a.txt' }),
  })
  assert.deepEqual(received.at(-1), { type: 'finish', reason: { kind: 'tool-calls' } })
})

test('keeps two tool calls apart when the server omits ids and indexes', async () => {
  const received = await chunks([
    line({
      message: {
        tool_calls: [
          { function: { name: 'one', arguments: { a: 1 } } },
          { function: { name: 'two', arguments: { b: 2 } } },
        ],
      },
      done: true,
      done_reason: 'tool_calls',
    }),
  ])

  const ends = received.filter((chunk) => chunk.type === 'block-end')
  assert.deepEqual(ends.map((chunk) => chunk.block.name), ['one', 'two'])
  assert.deepEqual(new Set(ends.map((chunk) => chunk.block.id)).size, 2)
})

test('reports a malformed line as a coded failure', async () => {
  await assert.rejects(
    () => chunks(['{"message":{}}\nnot json\n']),
    (error) => error.code === 'MALFORMED_RESPONSE',
  )
})

test('reports a provider error line as a coded failure', async () => {
  await assert.rejects(
    () => chunks([line({ error: 'model requires more system memory' })]),
    (error) => error.code === 'PROVIDER' && /system memory/u.test(error.message),
  )
})

test('rejects an empty completion instead of finishing cleanly', async () => {
  const received = await chunks([line({ done: true, done_reason: 'stop' })], { emptyResponseCode: 'LLM_EMPTY' })
  assert.deepEqual(received, [
    {
      type: 'finish',
      reason: { kind: 'error', failure: { message: received[0].reason.failure.message, code: 'LLM_EMPTY' } },
    },
  ])
})

test('maps length to max-tokens and unknown reasons to a failure', async () => {
  const short = await chunks([line({ message: { content: 'x' }, done: true, done_reason: 'length' })])
  assert.deepEqual(short.at(-1), { type: 'finish', reason: { kind: 'max-tokens' } })

  const odd = await chunks([line({ message: { content: 'x' }, done: true, done_reason: 'memory' })])
  assert.equal(odd.at(-1).reason.kind, 'error')
  assert.equal(odd.at(-1).reason.failure.code, 'PROVIDER')
})

test('serializes history, images and tool results for the wire', async () => {
  const messages = await wireMessages(
    [
      { role: 'system', content: [{ type: 'text', text: 'be brief' }] },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'what is this?' },
          { type: 'image', attachment: { name: 'shot.png', mediaType: 'image/png' } },
        ],
      },
      {
        role: 'assistant',
        content: [
          { type: 'reasoning', text: 'look first' },
          { type: 'text', text: 'reading' },
          { type: 'tool-call', id: 'call_1', name: 'read_file', arguments: '{"path":"a.txt"}' },
        ],
      },
      { role: 'tool', toolCallId: 'call_1', content: [{ type: 'text', text: 'file body' }] },
    ],
    {
      system: 'top-level system',
      readImage: async () => new Uint8Array([1, 2, 3]),
    },
  )

  assert.deepEqual(messages, [
    { role: 'system', content: 'top-level system' },
    { role: 'system', content: 'be brief' },
    { role: 'user', content: 'what is this?', images: ['AQID'] },
    {
      role: 'assistant',
      content: 'reading',
      thinking: 'look first',
      tool_calls: [{ id: 'call_1', function: { name: 'read_file', arguments: { path: 'a.txt' } } }],
    },
    { role: 'tool', content: 'file body' },
  ])
})

test('degrades an unreadable image to a placeholder instead of dropping the turn', async () => {
  const messages = await wireMessages(
    [{ role: 'user', content: [{ type: 'text', text: 'see' }, { type: 'image', attachment: { name: 'x.png' } }] }],
    { readImage: async () => { throw new Error('gone') } },
  )
  assert.equal(messages.length, 1)
  assert.equal(messages[0].images, undefined)
  assert.match(messages[0].content, /x\.png/u)
})

test('sends call controls and configured options inside body.options', () => {
  const body = buildChatBody(
    { model: 'llama3', temperature: 0.2, maxTokens: 256, stop: ['END'] },
    [{ role: 'user', content: 'hi' }],
    { capabilities: ['completion'] },
    { num_ctx: '8192', top_k: 20, temperature: '0.9', stop: 'a, b', num_predict: 64 },
  )

  assert.equal(body.model, 'llama3')
  assert.equal(body.stream, true)
  assert.deepEqual(body.options, {
    temperature: 0.9,
    num_predict: 64,
    stop: ['a', 'b'],
    num_ctx: 8192,
    top_k: 20,
  })
  assert.equal(body.tools, undefined)
  assert.equal(body.think, undefined)
})

test('declares tools only to models that accept them', () => {
  const tools = [{ name: 'read', description: 'read', parameters: { type: 'object', properties: {} } }]
  const withTools = buildChatBody({ model: 'm', tools }, [], { capabilities: ['tools'] }, {})
  assert.deepEqual(withTools.tools, [
    { type: 'function', function: { name: 'read', description: 'read', parameters: { type: 'object', properties: {} } } },
  ])

  const withoutTools = buildChatBody({ model: 'm', tools }, [], { capabilities: ['completion'] }, {})
  assert.equal(withoutTools.tools, undefined)

  const unknown = buildChatBody({ model: 'm', tools }, [], null, {})
  assert.equal(unknown.tools.length, 1)
})

test('maps the reasoning effort onto the native think field', () => {
  const thinking = { capabilities: ['thinking'] }
  assert.equal(buildChatBody({ model: 'm', reasoningEffort: 'high' }, [], thinking, {}).think, 'high')
  assert.equal(buildChatBody({ model: 'm', reasoningEffort: 'off' }, [], thinking, {}).think, false)
  assert.equal(buildChatBody({ model: 'm', reasoningEffort: 'on' }, [], thinking, {}).think, true)
  assert.equal(buildChatBody({ model: 'm', reasoningEffort: 'default' }, [], thinking, {}).think, undefined)
  // A manual setting is the fallback when the model has no native switch.
  assert.equal(buildChatBody({ model: 'm', reasoningEffort: 'high' }, [], { capabilities: ['completion'] }, { think: 'true' }).think, true)
  assert.equal(buildChatBody({ model: 'm' }, [], thinking, { think: false }).think, false)
})

test('keeps keep_alive at the body level and never inside options', () => {
  const body = buildChatBody({ model: 'm' }, [], null, { keep_alive: '10m', top_p: '0.8' })
  assert.equal(body.keep_alive, '10m')
  assert.deepEqual(body.options, { top_p: 0.8 })
})

/**
 * Translation of Ollama's /api/chat NDJSON stream into harness stream chunks.
 *
 * The harness speaks a small block protocol: `block-start` announces a block
 * index, deltas fill it, `block-end` carries its final content, and the stream
 * closes with `usage` (optional) and `finish`. Ollama, by contrast, resends a
 * whole `message` per line, with `thinking`, `content` and `tool_calls` able to
 * appear together.
 *
 * Dependency-free: protocol failures are thrown as `Error` values carrying a
 * stable `code`, and the adapter upgrades them to `LlmError`.
 *
 * @module lib/chunks
 */

/**
 * @typedef {{ type: string, [key: string]: unknown }} StreamChunkLike
 */

/**
 * Translate one parsed Ollama chat stream.
 *
 * @param {AsyncIterable<string>} lines NDJSON lines
 * @param {{ signal?: AbortSignal, emptyResponseCode?: string }} [options]
 * @returns {AsyncGenerator<StreamChunkLike>} harness stream chunks
 */
export async function* decodeOllamaStream(lines, options = {}) {
  const { signal, emptyResponseCode = 'EMPTY_RESPONSE' } = options

  /** @type {{ index: number, text: string } | null} */
  let reasoning = null
  /** @type {{ index: number, text: string } | null} */
  let text = null
  /** @type {Map<number, { index: number, id: string, name: string, arguments: string }>} */
  const calls = new Map()
  let nextIndex = 0
  /** @type {{ kind: string, failure?: { message: string, code: string } } | null} */
  let finish = null
  /** @type {{ inputTokens: number, outputTokens: number, totalTokens: number } | null} */
  let usage = null

  for await (const line of lines) {
    if (signal?.aborted) throw abortError()

    let chunk
    try {
      chunk = JSON.parse(line)
    } catch {
      throw protocolError(`Ollama sent malformed NDJSON: ${truncate(line)}`, 'MALFORMED_RESPONSE')
    }
    if (!chunk || typeof chunk !== 'object') {
      throw protocolError(`Ollama sent a non-object NDJSON line: ${truncate(line)}`, 'MALFORMED_RESPONSE')
    }
    if (typeof chunk.error === 'string' && chunk.error.length > 0) {
      throw protocolError(chunk.error, 'PROVIDER')
    }

    const message = chunk.message
    if (message && typeof message === 'object') {
      if (typeof message.thinking === 'string' && message.thinking.length > 0) {
        if (reasoning === null) {
          reasoning = { index: nextIndex++, text: '' }
          yield { type: 'block-start', index: reasoning.index, blockType: 'reasoning' }
        }
        reasoning.text += message.thinking
        yield { type: 'reasoning-delta', index: reasoning.index, text: message.thinking }
      }

      if (typeof message.content === 'string' && message.content.length > 0) {
        if (text === null) {
          text = { index: nextIndex++, text: '' }
          yield { type: 'block-start', index: text.index, blockType: 'text' }
        }
        text.text += message.content
        yield { type: 'text-delta', index: text.index, text: message.content }
      }

      if (Array.isArray(message.tool_calls)) {
        yield* toolCallDeltas(message.tool_calls, calls, () => nextIndex++)
      }
    }

    if (chunk.done === true) {
      finish = finishReasonOf(chunk.done_reason)
      usage = usageOf(chunk)
    }
  }

  // Close every block that was opened, in the order its index was assigned.
  if (reasoning !== null) {
    yield { type: 'block-end', index: reasoning.index, block: { type: 'reasoning', text: reasoning.text } }
  }
  if (text !== null) {
    yield { type: 'block-end', index: text.index, block: { type: 'text', text: text.text } }
  }
  for (const call of [...calls.values()].sort((left, right) => left.index - right.index)) {
    yield {
      type: 'block-end',
      index: call.index,
      block: { type: 'tool-call', id: call.id, name: call.name, arguments: call.arguments || '{}' },
    }
  }

  if (usage !== null) yield { type: 'usage', usage }

  const produced = (reasoning !== null ? 1 : 0) + (text !== null ? 1 : 0) + calls.size
  if (produced === 0) {
    yield {
      type: 'finish',
      reason: {
        kind: 'error',
        failure: {
          message: 'Ollama completed the response without producing any content',
          code: emptyResponseCode,
        },
      },
    }
    return
  }

  let reason = finish ?? { kind: 'stop' }
  // Some Ollama builds report done_reason "stop" even when they emitted tool
  // calls. The loop must resume with a tool turn, so a tool call wins.
  if (calls.size > 0 && reason.kind === 'stop') reason = { kind: 'tool-calls' }
  yield { type: 'finish', reason }
}

/**
 * Emit the deltas for one line's tool calls, accumulating them per call slot.
 *
 * @param {Array<any>} toolCalls Ollama tool calls
 * @param {Map<number, { index: number, id: string, name: string, arguments: string }>} calls state
 * @param {() => number} nextIndex block-index allocator
 * @returns {AsyncGenerator<StreamChunkLike>} deltas
 */
async function* toolCallDeltas(toolCalls, calls, nextIndex) {
  const bySlot = new Map()
  toolCalls.forEach((call, position) => {
    const slot = typeof call?.index === 'number' ? call.index : typeof call?.function?.index === 'number' ? call.function.index : position
    bySlot.set(slot, call)
  })

  for (const [slot, call] of bySlot) {
    const fn = call?.function
    if (!fn || typeof fn !== 'object') continue

    let state = calls.get(slot)
    if (!state) {
      state = {
        index: nextIndex(),
        id: typeof call.id === 'string' && call.id.length > 0 ? call.id : `ollama-call-${slot}`,
        name: typeof fn.name === 'string' ? fn.name : '',
        arguments: '',
      }
      calls.set(slot, state)
      yield { type: 'block-start', index: state.index, blockType: 'tool-call' }
    } else {
      if (typeof call.id === 'string' && call.id.length > 0) state.id = call.id
      if (typeof fn.name === 'string' && fn.name.length > 0) state.name = fn.name
    }

    const delta = argumentsDelta(fn.arguments, state.arguments)
    if (delta.length === 0) continue

    state.arguments += delta
    yield {
      type: 'tool-call-delta',
      index: state.index,
      id: state.id,
      ...(state.name ? { name: state.name } : {}),
      argumentsDelta: delta,
    }
  }
}

/**
 * Text to append for one arguments value. Ollama sends the full JSON object on
 * every line, so the emitted delta is the part that is not already buffered.
 *
 * @param {unknown} value raw arguments
 * @param {string} buffered arguments accumulated so far
 * @returns {string} delta
 */
function argumentsDelta(value, buffered) {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') {
    // A string form may still be a complete JSON document (Ollama does not
    // stream argument fragments), so treat it the same way.
    return value === buffered ? '' : value.slice(buffered.length)
  }
  const serialized = JSON.stringify(value)
  return serialized === buffered ? '' : serialized.slice(buffered.length)
}

/**
 * Map Ollama's `done_reason` onto a harness finish reason.
 *
 * @param {unknown} reason Ollama done reason
 * @returns {{ kind: string, failure?: { message: string, code: string } }} finish reason
 */
export function finishReasonOf(reason) {
  switch (reason) {
    case 'stop':
      return { kind: 'stop' }
    case 'length':
      return { kind: 'max-tokens' }
    case 'tool_calls':
    case 'tool-calls':
      return { kind: 'tool-calls' }
    case undefined:
    case null:
    case '':
      // Older builds omit the field entirely on a clean stop.
      return { kind: 'stop' }
    default: {
      const label = String(reason)
      return {
        kind: 'error',
        failure: { message: `Ollama stopped with reason "${label}"`, code: 'PROVIDER' },
      }
    }
  }
}

/**
 * Token usage reported on the final chunk.
 *
 * @param {Record<string, any>} chunk final chunk
 * @returns {{ inputTokens: number, outputTokens: number, totalTokens: number } | null} usage
 */
function usageOf(chunk) {
  const input = countOf(chunk.prompt_eval_count)
  const output = countOf(chunk.eval_count)
  if (input === undefined && output === undefined) return null
  const inputTokens = input ?? 0
  const outputTokens = output ?? 0
  return { inputTokens, outputTokens, totalTokens: inputTokens + outputTokens }
}

function countOf(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

/**
 * @param {string} message failure text
 * @param {string} code failure code
 * @returns {Error} protocol error carrying its code
 */
function protocolError(message, code) {
  const error = new Error(message)
  error.code = code
  return error
}

function abortError() {
  return protocolError('Ollama request aborted', 'ABORTED')
}

function truncate(value) {
  const text = String(value)
  return text.length > 160 ? `${text.slice(0, 160)}…` : text
}

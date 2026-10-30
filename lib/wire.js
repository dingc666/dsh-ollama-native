/**
 * Harness history → Ollama /api/chat wire messages.
 *
 * Ollama's wire format is deliberately simple: `role`, `content`, optional
 * `thinking`, optional `images` (base64), and `tool_calls` for an assistant.
 * Tool results are their own `role: "tool"` messages.
 *
 * Dependency-free so it can be unit-tested without a harness.
 *
 * @module lib/wire
 */

/**
 * Concatenate the text blocks of one message.
 *
 * @param {readonly any[] | undefined} blocks content blocks
 * @returns {string} text
 */
export function textOf(blocks) {
  if (!Array.isArray(blocks)) return ''
  let text = ''
  for (const block of blocks) {
    if (block?.type === 'text' && typeof block.text === 'string') text += block.text
  }
  return text
}

/**
 * Base64-encode image bytes for inline transmission.
 *
 * @param {Uint8Array} bytes image bytes
 * @returns {string} base64
 */
export function base64Of(bytes) {
  return Buffer.from(bytes).toString('base64')
}

/**
 * Placeholder used when an image cannot be read, so the turn still carries the
 * fact that an image was attached.
 *
 * @param {any} attachment image attachment reference
 * @returns {string} placeholder
 */
export function imagePlaceholder(attachment) {
  const name = typeof attachment?.name === 'string' && attachment.name.length > 0 ? attachment.name : 'image'
  const type = typeof attachment?.mediaType === 'string' ? ` ${attachment.mediaType}` : ''
  return `[${name}${type} could not be attached to this Ollama request]`
}

/**
 * One assistant turn, including tool calls it made.
 *
 * @param {any} message harness assistant message
 * @returns {Record<string, unknown>} wire message
 */
export function assistantWire(message) {
  const parts = []
  const thinking = []
  const toolCalls = []

  for (const block of Array.isArray(message?.content) ? message.content : []) {
    switch (block?.type) {
      case 'text':
        if (typeof block.text === 'string') parts.push(block.text)
        break
      case 'reasoning':
        if (typeof block.text === 'string') thinking.push(block.text)
        break
      case 'tool-call':
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : undefined,
          function: {
            name: block.name,
            arguments: parseArguments(block.arguments),
          },
        })
        break
      default:
        // images, files, tool-addition/removal: nothing Ollama consumes here.
        break
    }
  }

  const wire = { role: 'assistant', content: parts.join('') }
  if (thinking.length > 0) wire.thinking = thinking.join('')
  if (toolCalls.length > 0) wire.tool_calls = toolCalls
  return wire
}

/**
 * Convert harness history into wire messages, resolving images through the
 * attachment store when one is available.
 *
 * @param {readonly any[]} messages harness history
 * @param {{
 *   system?: string,
 *   readImage?: (attachment: any, signal?: AbortSignal) => Promise<Uint8Array | undefined>,
 *   signal?: AbortSignal,
 * }} [options]
 * @returns {Promise<Array<Record<string, unknown>>>} wire messages
 */
export async function wireMessages(messages, options = {}) {
  const { system, readImage, signal } = options
  const wire = []

  if (typeof system === 'string' && system.length > 0) {
    wire.push({ role: 'system', content: system })
  }

  for (const message of Array.isArray(messages) ? messages : []) {
    switch (message?.role) {
      case 'system':
      case 'developer':
        wire.push({ role: 'system', content: textOf(message.content) })
        break
      case 'assistant':
        wire.push(assistantWire(message))
        break
      case 'tool':
        // A tool result is a whole message in the harness model, and a whole
        // message in Ollama's protocol too.
        wire.push({ role: 'tool', content: textOf(message.content) || '(no output)' })
        break
      default:
        await appendUserTurn(wire, message, { readImage, signal })
        break
    }
  }

  return wire
}

async function appendUserTurn(wire, message, { readImage, signal }) {
  const parts = []
  const images = []

  for (const block of Array.isArray(message?.content) ? message.content : []) {
    switch (block?.type) {
      case 'text':
        if (typeof block.text === 'string') parts.push(block.text)
        break
      case 'image': {
        const bytes = await readImageBytes(block, readImage, signal)
        if (bytes) images.push(base64Of(bytes))
        else parts.push(imagePlaceholder(block.attachment))
        break
      }
      default:
        // Files are projected to deterministic handle text by the harness, so no
        // raw file bytes reach an adapter.
        break
    }
  }

  const text = parts.join('')
  wire.push({ role: 'user', content: text, ...(images.length > 0 ? { images } : {}) })
}

async function readImageBytes(block, readImage, signal) {
  if (typeof readImage !== 'function' || !block?.attachment) return undefined
  try {
    return await readImage(block.attachment, signal)
  } catch {
    return undefined
  }
}

function parseArguments(value) {
  if (value === null || value === undefined) return {}
  if (typeof value === 'object') return value
  try {
    const parsed = JSON.parse(String(value))
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
  } catch {
    return {}
  }
}

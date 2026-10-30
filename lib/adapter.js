/**
 * The harness `LlmAdapter` for Ollama's native /api/chat protocol.
 *
 * One adapter instance serves the provider route named by the configuration.
 * Options are read through a getter, so a volatile settings edit takes effect on
 * the next call without remounting the row.
 *
 * @module lib/adapter
 */

import { buildChatBody } from './body.js'
import { decodeOllamaStream } from './chunks.js'
import { EMPTY_RESPONSE_CODE, LlmAdapter, LlmError } from './harness.js'
import {
  capabilitiesOf,
  contextLengthOfShow,
  contextLengthOfTags,
  fetchShow,
  fetchTags,
  openChatStream,
} from './client.js'
import { splitNdjson } from './ndjson.js'
import { wireMessages } from './wire.js'

/** Model metadata is stable for a while; misses expire too, so a pull shows up. */
const FACTS_TTL_MS = 5 * 60_000
/** The effort id that means "leave the server's own thinking default alone". */
const DEFAULT_EFFORT = 'default'

export class OllamaAdapter extends LlmAdapter {
  /** @type {() => ReturnType<import('./config.js').resolveOptions>} */
  #options

  /** @type {() => { readImage?: Function } | undefined} */
  #attachments

  /** @type {Map<string, { expiresAt: number, value: ModelFacts | null }>} */
  #facts = new Map()

  /**
   * @param {() => any} options current resolved options
   * @param {() => any} attachments optional attachment store accessor
   */
  constructor(options, attachments) {
    super()
    this.#options = options
    this.#attachments = attachments ?? (() => undefined)
  }

  /** {@inheritDoc} */
  providerInfo(provider) {
    return { id: provider, name: this.#options().displayName }
  }

  /** {@inheritDoc} */
  async listModels(provider) {
    const { baseURL, requestTimeoutMs } = this.#options()
    let entries = []
    try {
      entries = await fetchTags(baseURL, undefined, requestTimeoutMs)
    } catch {
      // An unreachable server is a model list with nothing in it, not a failure
      // of the settings surface that asks for it.
      return []
    }
    return entries
      .filter((entry) => typeof entry?.name === 'string' && entry.name.length > 0)
      .map((entry) => {
        const modalities = modalitiesOf(capabilitiesOf(entry.capabilities))
        const description = describeModel(entry)
        return {
          provider,
          id: entry.name,
          name: entry.name,
          ...(description ? { description } : {}),
          inputModalities: modalities,
        }
      })
  }

  /** {@inheritDoc} */
  async resolveModel(provider, model, signal) {
    const options = this.#options()
    const catalog = options.models.find((entry) => entry.id === model)
    const facts = await this.#factsOf(options.baseURL, model, signal)

    const contextWindow =
      catalog?.contextWindow ?? facts?.contextLength ?? options.defaultContextWindow
    const reasoning = reasoningInfoOf(facts?.thinking)

    return {
      provider,
      id: model,
      name: catalog?.name ?? model,
      ...(catalog?.description ? { description: catalog.description } : {}),
      inputModalities:
        catalog?.inputModalities && catalog.inputModalities.length > 0
          ? catalog.inputModalities
          : modalitiesOf(facts?.capabilities),
      context: { contextWindow },
      defaultMaxTokens: catalog?.maxTokens ?? options.maxTokens,
      ...(reasoning ? { reasoning } : {}),
    }
  }

  /** {@inheritDoc} */
  async *stream(options) {
    const settings = this.#options()
    const facts = await this.#factsOf(settings.baseURL, options.model, options.signal)
    const messages = await wireMessages(options.messages, {
      system: options.system,
      readImage: this.#readImage,
      signal: options.signal,
    })
    const body = buildChatBody(options, messages, facts, settings.params)

    const response = await openChatStream(settings.baseURL, body, options.signal)
    if (!response.body) {
      throw new LlmError('Ollama returned a chat response without a body', 'EMPTY_RESPONSE')
    }

    const guard = createIdleGuard(settings.streamIdleTimeoutMs, options.signal)
    try {
      for await (const chunk of decodeOllamaStream(pulsing(splitNdjson(response.body), guard.pulse), {
        signal: guard.signal,
        emptyResponseCode: EMPTY_RESPONSE_CODE,
      })) {
        guard.pulse()
        yield chunk
      }
    } catch (error) {
      throw upgradeFailure(error, options.signal, guard, settings.streamIdleTimeoutMs)
    } finally {
      guard.dispose()
    }
  }

  /**
   * Resolve one model's server-side capabilities, cached briefly.
   *
   * `/api/tags` is authoritative for `capabilities` and usually carries
   * `details.context_length`; `/api/show` fills in what it does not, plus the
   * `thinking` descriptor. An unreachable server caches nothing and simply
   * leaves the request to fail loudly when it is attempted.
   *
   * @param {string} baseURL normalized base URL
   * @param {string} model model id
   * @param {AbortSignal} [signal] caller cancellation
   * @returns {Promise<ModelFacts | null>} facts, when the server reported any
   */
  async #factsOf(baseURL, model, signal) {
    const key = `${baseURL}\u0000${model}`
    const cached = this.#facts.get(key)
    const now = Date.now()
    if (cached && cached.expiresAt > now) return cached.value

    const { requestTimeoutMs } = this.#options()
    let entries = []
    try {
      entries = await fetchTags(baseURL, signal, requestTimeoutMs)
    } catch {
      // Fall through to /api/show.
    }

    const entry = entries.find((candidate) => candidate?.name === model) ?? entries.find((candidate) => candidate?.model === model)
    let capabilities = capabilitiesOf(entry?.capabilities)
    let contextLength = entry ? contextLengthOfTags(entry) : undefined
    let thinking

    if (!entry || capabilities?.includes('thinking')) {
      try {
        const show = await fetchShow(baseURL, model, signal, requestTimeoutMs)
        capabilities ??= capabilitiesOf(show?.capabilities)
        contextLength ??= contextLengthOfShow(show)
        thinking = show?.thinking
      } catch {
        // Details are advisory: a model still streams without them.
      }
    }

    const value = entry || capabilities || contextLength ? { capabilities, contextLength, thinking } : null
    this.#facts.set(key, { expiresAt: now + FACTS_TTL_MS, value })
    return value
  }

  /** @param {any} attachment attachment reference @param {AbortSignal} [signal] cancellation */
  #readImage = async (attachment, signal) => {
    const store = this.#attachments()
    if (!store || typeof store.readImage !== 'function') return undefined
    const stored = await store.readImage(attachment, signal)
    const bytes = stored?.data ?? stored?.bytes ?? stored
    return bytes instanceof Uint8Array ? bytes : undefined
  }
}

/**
 * @typedef {{
 *   capabilities?: string[],
 *   contextLength?: number,
 *   thinking?: { values?: unknown[], default?: unknown },
 * }} ModelFacts
 */

function modalitiesOf(capabilities) {
  return Array.isArray(capabilities) && capabilities.includes('vision') ? ['text', 'image'] : ['text']
}

function describeModel(entry) {
  const details = entry?.details
  if (!details || typeof details !== 'object') return undefined
  const parts = []
  if (typeof details.parameter_size === 'string' && details.parameter_size) parts.push(details.parameter_size)
  if (typeof details.quantization_level === 'string' && details.quantization_level) parts.push(details.quantization_level)
  if (typeof details.family === 'string' && details.family) parts.push(details.family)
  return parts.length > 0 ? parts.join(' · ') : undefined
}

/**
 * Reasoning-effort metadata from Ollama's per-model `thinking` descriptor.
 *
 * @param {{ values?: unknown[], default?: unknown } | undefined} thinking descriptor
 * @returns {{ efforts: Array<{ id: string, name: string }>, defaultEffort: string } | undefined} metadata
 */
function reasoningInfoOf(thinking) {
  const values = thinking?.values
  if (!Array.isArray(values) || values.length === 0) return undefined

  const efforts = [{ id: DEFAULT_EFFORT, name: 'Server default' }]
  if (values.every((value) => typeof value === 'boolean')) {
    efforts.push({ id: 'off', name: 'Off' }, { id: 'on', name: 'On' })
  } else {
    for (const value of values) {
      const id = String(value)
      efforts.push({ id, name: id.charAt(0).toUpperCase() + id.slice(1) })
    }
  }
  return { efforts, defaultEffort: DEFAULT_EFFORT }
}

async function* pulsing(lines, pulse) {
  for await (const line of lines) {
    pulse()
    yield line
  }
}

/**
 * Idle watchdog: abort a stream that stops producing bytes.
 *
 * @param {number} timeoutMs idle budget
 * @param {AbortSignal} [upstream] caller cancellation
 */
function createIdleGuard(timeoutMs, upstream) {
  const controller = new AbortController()
  let timer
  let timedOut = false

  const arm = () => {
    if (timer !== undefined) clearTimeout(timer)
    timer = setTimeout(() => {
      timedOut = true
      controller.abort(new Error(`Ollama sent nothing for ${timeoutMs}ms`))
    }, timeoutMs)
    if (typeof timer?.unref === 'function') timer.unref()
  }

  const signal = upstream ? AbortSignal.any([upstream, controller.signal]) : controller.signal
  arm()

  return {
    signal,
    pulse: arm,
    get timedOut() {
      return timedOut
    },
    dispose() {
      if (timer !== undefined) clearTimeout(timer)
      timer = undefined
    },
  }
}

/**
 * Turn whatever interrupted the stream into the failure the harness understands:
 * protocol codes survive, an idle watchdog reports TIMEOUT, a caller abort
 * reports ABORTED, everything else is TRANSPORT.
 */
function upgradeFailure(error, upstream, guard, idleTimeoutMs) {
  if (error instanceof LlmError) return error
  if (guard.timedOut) {
    return new LlmError(`Ollama stream went idle for ${idleTimeoutMs}ms`, 'TIMEOUT', { cause: error })
  }
  if (upstream?.aborted) {
    return new LlmError('Ollama request aborted', 'ABORTED', { cause: error })
  }
  const code = typeof error?.code === 'string' && error.code.length > 0 ? error.code : 'TRANSPORT'
  const message = error instanceof Error ? error.message : String(error)
  return new LlmError(message, code, { cause: error })
}

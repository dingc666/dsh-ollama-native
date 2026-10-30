/**
 * Configuration schema and normalization for the Ollama native provider.
 *
 * Every user-facing field is `.volatile()`: the harness projects the Loader's
 * own configuration into the Plugins page, and a volatile field is committed
 * into the reference the running plugin already holds, so an edit reaches the
 * adapter without remounting the row. `providerId` is deliberately NOT
 * volatile — it is the route the adapter registers on, which is fixed for the
 * lifetime of one mount.
 *
 * @module lib/config
 */

import { isVolatile, z } from './harness.js'

export const DEFAULT_BASE_URL = 'http://127.0.0.1:11434'
export const DEFAULT_DISPLAY_NAME = 'Ollama'
export const DEFAULT_PROVIDER_ID = 'ollama'
/** Ollama's own default context; used only when a model reports nothing. */
export const DEFAULT_CONTEXT_WINDOW = 131_072
export const DEFAULT_MAX_TOKENS = 16384
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000

/** setTimeout cannot exceed this, and neither may the idle watchdog. */
const MAX_TIMER_DELAY_MS = 2 ** 31 - 1

const MODALITIES = new Set(['text', 'image'])

export const Config = z.object({
  /** Ollama server origin, optionally with a path prefix (no trailing slash). */
  baseURL: z.string().default(DEFAULT_BASE_URL).description('Ollama server URL').volatile(),
  /** Provider route registered with `ctx.llm`; fixed until the row remounts. */
  providerId: z.string().default(DEFAULT_PROVIDER_ID).description('Provider route id'),
  /** Name shown in the model picker. */
  displayName: z.string().default(DEFAULT_DISPLAY_NAME).description('Provider display name').volatile(),
  /** Context window for models that report none of their own. */
  defaultContextWindow: z
    .number()
    .min(1)
    .default(DEFAULT_CONTEXT_WINDOW)
    .description('Fallback context window')
    .volatile(),
  /** Default output-token ceiling when the caller sends none. */
  maxTokens: z.number().min(1).default(DEFAULT_MAX_TOKENS).description('Default max output tokens').volatile(),
  /** Abort a generation that produces no bytes for this long. */
  streamIdleTimeoutMs: z
    .number()
    .min(1)
    .max(MAX_TIMER_DELAY_MS)
    .default(DEFAULT_STREAM_IDLE_TIMEOUT_MS)
    .description('Stream idle timeout (ms)')
    .volatile(),
  /** Deadline for the small metadata calls (/api/tags, /api/show, /api/version). */
  requestTimeoutMs: z
    .number()
    .min(1)
    .max(MAX_TIMER_DELAY_MS)
    .default(10_000)
    .description('Metadata request timeout (ms)')
    .volatile(),
  /**
   * Extra catalog entries merged over what /api/tags reports. Each entry wins
   * over the server's own metadata for that model id.
   */
  models: z
    .array(
      z.object({
        id: z.string().required(),
        name: z.string(),
        description: z.string(),
        contextWindow: z.number().min(1),
        maxTokens: z.number().min(1),
        inputModalities: z.array(z.string()).min(1).default(['text']),
      }),
    )
    .default([])
    .description('Static model catalog overlay')
    .volatile(),
  /**
   * Ollama generation options. Keys are Ollama's own option names and are sent
   * inside the request's `options` object; `keep_alive` and `think` are body
   * fields and are handled as such.
   */
  params: z
    .object({
      temperature: z.any(),
      top_p: z.any(),
      top_k: z.any(),
      min_p: z.any(),
      typical_p: z.any(),
      tfs_z: z.any(),
      seed: z.any(),
      repeat_penalty: z.any(),
      repeat_last_n: z.any(),
      frequency_penalty: z.any(),
      presence_penalty: z.any(),
      mirostat: z.any(),
      mirostat_tau: z.any(),
      mirostat_eta: z.any(),
      num_ctx: z.any(),
      num_predict: z.any(),
      num_thread: z.any(),
      stop: z.any(),
      think: z.any(),
      keep_alive: z.any(),
    })
    .default({})
    .description('Ollama generation options')
    .volatile(),
})

/**
 * Replace each volatile reference with the value currently behind it. A schema
 * field the harness marks volatile resolves to a reference rather than a plain
 * value, because edits are committed in place.
 *
 * @param {Record<string, unknown>} config parsed plugin config
 * @returns {Record<string, unknown>} plain values
 */
export function plainConfig(config) {
  const plain = {}
  for (const [key, value] of Object.entries(config ?? {})) {
    plain[key] = isVolatile(value) ? value.get() : value
  }
  return plain
}

/**
 * Normalize a configured server URL: trim, validate, drop trailing slashes.
 *
 * @param {unknown} value user-configured URL
 * @returns {string} normalized base URL
 */
export function normalizeBaseURL(value) {
  const raw = typeof value === 'string' && value.trim().length > 0 ? value.trim() : DEFAULT_BASE_URL
  let url
  try {
    url = new URL(raw)
  } catch {
    throw new Error(`dsh-ollama-native: baseURL is not a valid URL: ${JSON.stringify(value)}`)
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error(`dsh-ollama-native: baseURL must be http(s): ${JSON.stringify(value)}`)
  }
  const path = url.pathname.replace(/\/+$/u, '')
  return `${url.origin}${path}`
}

/**
 * Turn a raw Config snapshot into the plain options the adapter reads.
 *
 * @param {Record<string, unknown>} config parsed plugin config
 * @returns {{
 *   providerId: string,
 *   displayName: string,
 *   baseURL: string,
 *   defaultContextWindow: number,
 *   maxTokens: number,
 *   streamIdleTimeoutMs: number,
 *   requestTimeoutMs: number,
 *   models: Array<Record<string, unknown>>,
 *   params: Record<string, unknown>,
 * }} resolved options
 */
export function resolveOptions(config) {
  const plain = plainConfig(config)

  const providerId = text(plain.providerId) ?? DEFAULT_PROVIDER_ID
  const displayName = text(plain.displayName) ?? DEFAULT_DISPLAY_NAME
  const baseURL = normalizeBaseURL(plain.baseURL)
  const defaultContextWindow = positiveInteger(plain.defaultContextWindow) ?? DEFAULT_CONTEXT_WINDOW
  const maxTokens = positiveInteger(plain.maxTokens) ?? DEFAULT_MAX_TOKENS
  const streamIdleTimeoutMs = timerDelay(plain.streamIdleTimeoutMs) ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS
  const requestTimeoutMs = timerDelay(plain.requestTimeoutMs) ?? 10_000

  const models = []
  for (const entry of Array.isArray(plain.models) ? plain.models : []) {
    const id = text(entry?.id)
    if (!id) continue
    const declared = Array.isArray(entry.inputModalities) ? entry.inputModalities : []
    models.push({
      id,
      name: text(entry.name) ?? id,
      ...(text(entry.description) ? { description: text(entry.description) } : {}),
      ...(positiveInteger(entry.contextWindow) ? { contextWindow: positiveInteger(entry.contextWindow) } : {}),
      ...(positiveInteger(entry.maxTokens) ? { maxTokens: positiveInteger(entry.maxTokens) } : {}),
      inputModalities: declared.filter((modality) => MODALITIES.has(modality)),
    })
  }

  const params = plain.params && typeof plain.params === 'object' ? plain.params : {}

  return {
    providerId,
    displayName,
    baseURL,
    defaultContextWindow,
    maxTokens,
    streamIdleTimeoutMs,
    requestTimeoutMs,
    models,
    params,
  }
}

function text(value) {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : undefined
}

function positiveInteger(value) {
  const number = typeof value === 'string' ? Number(value) : value
  return typeof number === 'number' && Number.isFinite(number) && number > 0 ? Math.floor(number) : undefined
}

function timerDelay(value) {
  const number = positiveInteger(value)
  return number !== undefined && number <= MAX_TIMER_DELAY_MS ? number : undefined
}

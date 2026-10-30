/**
 * HTTP access to one Ollama server.
 *
 * Everything here is transport only: no harness message shapes, no streaming
 * decoding. Failures become `LlmError` with the code the retry policy and the
 * transcript already understand.
 *
 * @module lib/client
 */

import { LlmError, attributionHeaders } from './harness.js'

/**
 * @param {string} baseURL normalized base URL
 * @param {string} path API path starting with "/"
 * @returns {string} absolute URL
 */
export function endpointOf(baseURL, path) {
  return `${baseURL}${path}`
}

/**
 * Map an HTTP status onto the stable failure code used by the harness.
 *
 * @param {number} status response status
 * @returns {string} failure code
 */
export function failureCodeOf(status) {
  if (status === 400) return 'INVALID_REQUEST'
  if (status === 401 || status === 403) return 'AUTH'
  if (status === 404) return 'NOT_FOUND'
  if (status === 429) return 'RATE_LIMIT'
  if (status >= 500) return 'SERVER'
  return `HTTP_${status}`
}

/**
 * Build the `LlmError` for a non-2xx Ollama response, preferring the server's
 * own `error` text.
 *
 * @param {Response} response failed response
 * @param {string} url request URL, echoed in the fallback message
 * @returns {Promise<LlmError>} the normalized failure
 */
export async function failureFromResponse(response, url) {
  let message = `Ollama request failed with HTTP ${response.status} (${url})`
  try {
    const body = await response.json()
    if (body && typeof body.error === 'string' && body.error.length > 0) message = body.error
  } catch {
    // An unparseable error body is not itself a new failure: status decides.
  }
  return new LlmError(message, failureCodeOf(response.status), { status: response.status })
}

function headers(accept) {
  return { accept, ...attributionHeaders() }
}

/**
 * Combine a caller signal with a deadline, when the runtime supports it.
 *
 * @param {AbortSignal | undefined} signal caller cancellation
 * @param {number} timeoutMs deadline
 * @returns {AbortSignal} signal for the request
 */
function withDeadline(signal, timeoutMs) {
  const deadline = AbortSignal.timeout(timeoutMs)
  if (!signal) return deadline
  return typeof AbortSignal.any === 'function' ? AbortSignal.any([signal, deadline]) : signal
}

/**
 * GET one JSON document.
 *
 * @param {string} baseURL normalized base URL
 * @param {string} path API path
 * @param {{ signal?: AbortSignal, timeoutMs?: number }} options request options
 * @returns {Promise<any>} parsed body
 */
async function getJson(baseURL, path, { signal, timeoutMs = 10_000 } = {}) {
  const url = endpointOf(baseURL, path)
  let response
  try {
    response = await fetch(url, {
      method: 'GET',
      headers: headers('application/json'),
      signal: withDeadline(signal, timeoutMs),
    })
  } catch (error) {
    throw transportError(error, url, signal)
  }
  if (!response.ok) throw await failureFromResponse(response, url)
  return response.json()
}

/**
 * POST one JSON document and return its parsed body.
 *
 * @param {string} baseURL normalized base URL
 * @param {string} path API path
 * @param {unknown} body request body
 * @param {{ signal?: AbortSignal, timeoutMs?: number }} options request options
 * @returns {Promise<any>} parsed body
 */
async function postJson(baseURL, path, body, { signal, timeoutMs = 10_000 } = {}) {
  const url = endpointOf(baseURL, path)
  let response
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { ...headers('application/json'), 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: withDeadline(signal, timeoutMs),
    })
  } catch (error) {
    throw transportError(error, url, signal)
  }
  if (!response.ok) throw await failureFromResponse(response, url)
  return response.json()
}

/**
 * Open a streaming /api/chat response. The caller owns body decoding.
 *
 * @param {string} baseURL normalized base URL
 * @param {unknown} body request body
 * @param {AbortSignal | undefined} signal caller cancellation
 * @returns {Promise<Response>} the streaming response
 */
export async function openChatStream(baseURL, body, signal) {
  const url = endpointOf(baseURL, '/api/chat')
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { ...headers('application/x-ndjson'), 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
    if (!response.ok) throw await failureFromResponse(response, url)
    return response
  } catch (error) {
    if (error instanceof LlmError) throw error
    throw transportError(error, url, signal)
  }
}

/**
 * Installed models, as reported by /api/tags.
 *
 * @param {string} baseURL normalized base URL
 * @param {AbortSignal} [signal] caller cancellation
 * @param {number} [timeoutMs] deadline
 * @returns {Promise<Array<Record<string, any>>>} tag entries
 */
export async function fetchTags(baseURL, signal, timeoutMs) {
  const body = await getJson(baseURL, '/api/tags', { signal, timeoutMs })
  return Array.isArray(body?.models) ? body.models : []
}

/**
 * One model's details, as reported by /api/show.
 *
 * @param {string} baseURL normalized base URL
 * @param {string} model model id
 * @param {AbortSignal} [signal] caller cancellation
 * @param {number} [timeoutMs] deadline
 * @returns {Promise<Record<string, any>>} show payload
 */
export async function fetchShow(baseURL, model, signal, timeoutMs) {
  return postJson(baseURL, '/api/show', { model }, { signal, timeoutMs })
}

/**
 * A model's context length from one /api/tags entry.
 *
 * @param {Record<string, any>} entry tag entry
 * @returns {number | undefined} context length
 */
export function contextLengthOfTags(entry) {
  const length = entry?.details?.context_length
  return typeof length === 'number' && Number.isFinite(length) && length > 0 ? length : undefined
}

/**
 * A model's context length from one /api/show payload, which reports it inside
 * `model_info` under a family-prefixed key.
 *
 * @param {Record<string, any>} show show payload
 * @returns {number | undefined} context length
 */
export function contextLengthOfShow(show) {
  const info = show?.model_info
  if (!info || typeof info !== 'object') return undefined
  for (const [key, value] of Object.entries(info)) {
    if (key !== 'context_length' && !key.endsWith('.context_length')) continue
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) return value
  }
  return undefined
}

/**
 * Capabilities reported for a model, if any.
 *
 * @param {unknown} capabilities raw capabilities
 * @returns {string[] | undefined} capabilities
 */
export function capabilitiesOf(capabilities) {
  return Array.isArray(capabilities) ? capabilities.filter((entry) => typeof entry === 'string') : undefined
}

function transportError(error, url, signal) {
  if (signal?.aborted) return new LlmError('Ollama request aborted', 'ABORTED', { cause: error })
  if (error instanceof LlmError) return error
  const detail = error instanceof Error ? error.message : String(error)
  return new LlmError(`Ollama request to ${url} failed: ${detail}`, 'TRANSPORT', { cause: error })
}

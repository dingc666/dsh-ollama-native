/**
 * Request-body construction for /api/chat.
 *
 * Kept pure and separate from the adapter: this is where sampling controls, the
 * configured Ollama options, tool declarations and the native thinking switch
 * are reconciled, and it is the part most worth testing without a server.
 *
 * @module lib/body
 */

/** The effort id that means "leave the server's own thinking default alone". */
const DEFAULT_EFFORT = 'default'
/** Body fields whose value is taken verbatim from the plugin configuration. */
const BODY_LEVEL = new Set(['think', 'keep_alive'])

/**
 * Build the /api/chat request body.
 *
 * Precedence: an explicit call control (temperature/maxTokens/stop) is applied
 * first, then the configured Ollama options override it, because the plugin
 * configuration is the user's deliberate choice for this provider.
 *
 * @param {any} options harness generation options
 * @param {Array<Record<string, unknown>>} messages wire messages
 * @param {{ capabilities?: string[], thinking?: any } | null} facts model facts
 * @param {Record<string, unknown>} params configured Ollama options
 * @returns {Record<string, unknown>} request body
 */
export function buildChatBody(options, messages, facts, params) {
  const body = { model: options.model, messages, stream: true }

  const tuning = {}
  if (typeof options.temperature === 'number') tuning.temperature = options.temperature
  if (typeof options.maxTokens === 'number' && options.maxTokens > 0) {
    tuning.num_predict = options.maxTokens
  }
  if (Array.isArray(options.stop) && options.stop.length > 0) {
    tuning.stop = stopList(options.stop)
  }

  for (const [key, raw] of Object.entries(params ?? {})) {
    if (raw === null || raw === undefined) continue
    if (BODY_LEVEL.has(key)) continue
    const value = coerce(raw)
    if (value === undefined) continue
    tuning[key] = key === 'stop' ? stopList(value) : value
  }
  if (Object.keys(tuning).length > 0) body.options = tuning

  const tools = toolsOf(options, facts)
  if (tools !== undefined) body.tools = tools

  const think = thinkOf(options, facts, params ?? {})
  if (think !== undefined) body.think = think

  const keepAlive = coerce(params?.keep_alive)
  if (typeof keepAlive === 'number' || typeof keepAlive === 'string') body.keep_alive = keepAlive

  return body
}

/**
 * Tool declarations, only for a model that can accept them.
 *
 * A model whose capabilities are unknown (server unreachable, older Ollama) is
 * allowed to try; a model that declares its capabilities must advertise tools.
 *
 * @param {any} options harness generation options
 * @param {{ capabilities?: string[] } | null} facts model facts
 * @returns {Array<Record<string, unknown>> | undefined} tools
 */
function toolsOf(options, facts) {
  const tools = options.tools
  if (!Array.isArray(tools) || tools.length === 0) return undefined
  const capabilities = facts?.capabilities
  if (Array.isArray(capabilities) && !capabilities.includes('tools')) return undefined
  return tools.map((tool) => ({
    type: 'function',
    function: { name: tool.name, description: tool.description, parameters: tool.parameters },
  }))
}

/**
 * The native thinking switch: a reasoning effort when the model declares one,
 * otherwise the explicit manual setting.
 *
 * @param {any} options harness generation options
 * @param {{ capabilities?: string[] } | null} facts model facts
 * @param {Record<string, unknown>} params configured Ollama options
 * @returns {boolean | string | undefined} value for the body's `think` field
 */
function thinkOf(options, facts, params) {
  const effort =
    options.reasoningEffort === null || options.reasoningEffort === undefined
      ? undefined
      : String(options.reasoningEffort)
  const supportsThinking = facts?.capabilities?.includes('thinking') === true

  if (supportsThinking && effort !== undefined && effort !== DEFAULT_EFFORT) {
    if (effort === 'off') return false
    if (effort === 'on') return true
    return effort
  }
  if (params.think === null || params.think === undefined) return undefined
  const manual = coerce(params.think)
  return typeof manual === 'boolean' || typeof manual === 'string' ? manual : undefined
}

/** Normalize an option the Plugins page may deliver as a string. */
function coerce(value) {
  if (typeof value === 'number' || typeof value === 'boolean') return value
  if (typeof value !== 'string') return value
  const text = value.trim()
  if (text.length === 0) return undefined
  if (text === 'true') return true
  if (text === 'false') return false
  if (/^-?\d+$/u.test(text)) return Number(text)
  if (/^-?(?:\d+\.\d*|\.\d+)$/u.test(text)) return Number(text)
  return text
}

/** A stop value is either an array or a comma/newline separated string. */
function stopList(value) {
  const values = Array.isArray(value) ? value : String(value).split(/[\n,]/u)
  return values.map((entry) => String(entry).trim()).filter((entry) => entry.length > 0)
}

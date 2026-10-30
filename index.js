/**
 * dsh-ollama-native — the Ollama provider row.
 *
 * Registers one `LlmAdapter` on the configured provider route plus the
 * configurable-provider directory entry and model discovery, so the harness can
 * list, pick and interrogate local Ollama models without an OpenAI-compatible
 * bridge.
 *
 * @module dsh-ollama-native
 */

import { OllamaAdapter } from './lib/adapter.js'
import { capabilitiesOf, contextLengthOfTags, fetchTags } from './lib/client.js'
import { Config, normalizeBaseURL, resolveOptions } from './lib/config.js'

export const name = 'llm-ollama-native'

/** The adapter registry this row feeds. */
export const inject = ['llm']

export { Config }

/**
 * @param {import('@deepseek-ai/cordis').Context} ctx plugin context
 * @param {Record<string, unknown>} config Loader config (volatile refs included)
 */
export function apply(ctx, config) {
  let lastGood

  /**
   * Resolve the live configuration. A volatile edit can be transiently invalid
   * while the field is being typed, so the last good snapshot keeps serving
   * instead of breaking an in-flight session.
   */
  const readOptions = () => {
    try {
      lastGood = resolveOptions(config)
      return lastGood
    } catch (error) {
      if (lastGood === undefined) throw error
      ctx.logger?.error?.(`${name}: ignoring an invalid configuration and keeping the last good one`)
      ctx.logger?.error?.(error)
      return lastGood
    }
  }

  const initial = readOptions()
  const provider = initial.providerId
  // The settings namespace is the Loader entry id carrying this plugin; it is
  // what the Settings surface holds when it edits or discovers this provider.
  const settingsNs = ctx.fiber?.entry?.options?.id ?? name

  const adapter = new OllamaAdapter(readOptions, () => ctx.get('attachments', false))

  ctx.llm.registerAdapter([provider], adapter)

  ctx.llm.registerConfigurableProviders([
    {
      provider,
      displayName: initial.displayName,
      settingsNs,
      settingsPath: [],
      defaultBaseURL: initial.baseURL,
    },
  ])

  ctx.llm.registerModelDiscovery(settingsNs, async (request, signal) => {
    const options = readOptions()
    const baseURL =
      typeof request?.baseURL === 'string' && request.baseURL.trim().length > 0
        ? normalizeBaseURL(request.baseURL)
        : options.baseURL
    const entries = await fetchTags(baseURL, signal, options.requestTimeoutMs)
    return entries
      .filter((entry) => typeof entry?.name === 'string' && entry.name.length > 0)
      .map((entry) => {
        const contextWindow = contextLengthOfTags(entry)
        return {
          id: entry.name,
          name: entry.name,
          ...(contextWindow ? { contextWindow } : {}),
          inputModalities: capabilitiesOf(entry.capabilities)?.includes('vision')
            ? ['text', 'image']
            : ['text'],
        }
      })
  })

  ctx.logger?.info?.(`${name}: provider "${provider}" → ${initial.baseURL}`)
}

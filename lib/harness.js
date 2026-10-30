/**
 * Access to the harness runtime packages from a plugin that lives outside the
 * dsh installation.
 *
 * A profile plugin is imported by path, so its bare imports resolve from the
 * plugin's own directory — which usually has no `node_modules`. The harness
 * packages therefore have to be located explicitly:
 *
 * 1. Ordinary bare resolution, which is right whenever the profile or the
 *    plugin's own dependencies provide the package.
 * 2. The runtime's own copy, imported by absolute file URL. This is the
 *    important case: it yields the *same* module instances the harness itself
 *    loaded, so `instanceof LlmError` / `extends LlmAdapter` keep working.
 *
 * The import happens at module scope, so a failure surfaces as a clear
 * activation error naming every path that was tried.
 *
 * @module lib/harness
 */

import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { pathToFileURL } from 'node:url'

/** Candidate roots of the dsh runtime, most specific first. */
function runtimeRoots() {
  const roots = []
  const push = (value) => {
    if (typeof value === 'string' && value.length > 0 && !roots.includes(value)) roots.push(value)
  }

  const resources = typeof process.resourcesPath === 'string' ? process.resourcesPath : ''
  if (resources.length > 0) {
    push(join(resources, 'app.asar', 'dsh'))
    push(join(resources, 'app', 'dsh'))
  }
  const executable = typeof process.execPath === 'string' ? dirname(process.execPath) : ''
  if (executable.length > 0) {
    push(join(executable, 'resources', 'app.asar', 'dsh'))
    push(join(executable, 'resources', 'app', 'dsh'))
  }
  return roots
}

/** Resolve a package directory's entry file the way Node would, by manifest. */
function entryOf(packageDir) {
  const manifestPath = join(packageDir, 'package.json')
  if (existsSync(manifestPath)) {
    try {
      const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'))
      const exported = manifest?.exports?.['.']
      const entry =
        typeof exported === 'string'
          ? exported
          : (exported?.import ?? exported?.default ?? manifest?.main)
      if (typeof entry === 'string' && entry.length > 0) {
        const candidate = join(packageDir, entry)
        if (existsSync(candidate)) return candidate
      }
    } catch {
      // A malformed manifest falls through to the conventional entries.
    }
  }
  for (const candidate of ['lib/index.js', 'lib/index.cjs', 'index.js', 'index.cjs']) {
    const path = join(packageDir, candidate)
    if (existsSync(path)) return path
  }
  return undefined
}

const loaded = new Map()

/**
 * Load one harness package, preferring the runtime's instance.
 *
 * @param {string} name package name
 * @returns {Promise<any>} the package namespace
 */
async function loadHarnessPackage(name) {
  const cached = loaded.get(name)
  if (cached !== undefined) return cached

  const tried = []
  let firstError

  try {
    const namespace = await import(name)
    loaded.set(name, namespace)
    return namespace
  } catch (error) {
    firstError = error
    tried.push(name)
  }

  for (const root of runtimeRoots()) {
    const entry = entryOf(join(root, 'node_modules', name))
    if (entry === undefined) continue
    tried.push(entry)
    try {
      const namespace = await import(pathToFileURL(entry).href)
      loaded.set(name, namespace)
      return namespace
    } catch (error) {
      firstError ??= error
    }
  }

  const detail = firstError instanceof Error ? firstError.message : String(firstError)
  throw new Error(
    `dsh-ollama-native: cannot load ${name}. Tried: ${tried.filter(Boolean).join(', ') || '(nothing)'}. ${detail}`,
  )
}

const llm = await loadHarnessPackage('@deepseek-ai/dsh-llm')

for (const required of ['LlmAdapter', 'LlmError']) {
  if (typeof llm?.[required] !== 'function') {
    throw new Error(`dsh-ollama-native: ${'@deepseek-ai/dsh-llm'} does not export ${required}`)
  }
}

/** The harness adapter base class (its defaults matter: only `stream` is abstract). */
export const LlmAdapter = llm.LlmAdapter

/** The harness failure type, so error codes and `instanceof` stay meaningful. */
export const LlmError = llm.LlmError

/** The harness code for "completed without content"; own fallback if absent. */
export const EMPTY_RESPONSE_CODE = llm.EMPTY_RESPONSE_CODE ?? 'EMPTY_RESPONSE'

/** Attribution headers, when the runtime offers them. */
export const attributionHeaders = typeof llm.attributionHeaders === 'function' ? llm.attributionHeaders : () => ({})

const schemastery = await loadHarnessPackage('@deepseek-ai/schemastery')

const schemasteryExports = schemastery?.default ?? schemastery

/** The schemastery instance the harness's Loader validates Config with. */
export const z = schemasteryExports?.default ?? schemasteryExports

const cosmokit = await loadHarnessPackage('@deepseek-ai/cosmokit')

const cosmokitExports = cosmokit?.isVolatile ? cosmokit : (cosmokit?.default ?? cosmokit)

/**
 * Whether a parsed Config field is a volatile reference.
 *
 * The runtime's cosmokit is preferred because a reference must be recognized by
 * the same instance that created it; the shape check is only a guard for exotic
 * resolutions.
 */
export const isVolatile =
  typeof cosmokitExports?.isVolatile === 'function'
    ? cosmokitExports.isVolatile
    : (value) =>
        typeof value?.get === 'function' &&
        Object.getOwnPropertySymbols(value ?? {}).some((symbol) => String(symbol).includes('volatile'))

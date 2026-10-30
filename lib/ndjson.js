/**
 * Newline-delimited JSON framing for Ollama's streaming endpoints.
 *
 * Dependency-free on purpose: this is the part of the adapter that a unit test
 * can drive without a harness or a server.
 *
 * @module lib/ndjson
 */

/**
 * Decode a byte stream into complete NDJSON lines.
 *
 * Partial lines are buffered until their newline arrives, `\r\n` is accepted,
 * blank lines are skipped, and a final line without a trailing newline is still
 * yielded. The reader lock is always released, including on abort.
 *
 * @param {ReadableStream<Uint8Array> | null | undefined} body response body
 * @returns {AsyncGenerator<string>} complete lines in delivery order
 */
export async function* splitNdjson(body) {
  if (!body) return
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let newline = buffer.indexOf('\n')
      while (newline !== -1) {
        const line = stripCarriageReturn(buffer.slice(0, newline))
        buffer = buffer.slice(newline + 1)
        if (line.trim().length > 0) yield line
        newline = buffer.indexOf('\n')
      }
    }
  } finally {
    reader.releaseLock()
  }
  buffer += decoder.decode()
  const tail = stripCarriageReturn(buffer)
  if (tail.trim().length > 0) yield tail
}

function stripCarriageReturn(line) {
  return line.endsWith('\r') ? line.slice(0, -1) : line
}

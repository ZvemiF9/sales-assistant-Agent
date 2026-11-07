import { createParser } from 'eventsource-parser'

const EVENTS = new Set(['status', 'text', 'products', 'sources', 'confirmation', 'done', 'error'])
const MAX_EVENT_BYTES = 1024 * 1024
const MAX_TOTAL_BYTES = 8 * 1024 * 1024
const MAX_EVENTS = 256

export class SSEError extends Error {
  constructor(code = 'agent_invalid_response') {
    super(code === 'request_aborted' ? 'Request stopped.' : 'Invalid stream.')
    this.name = 'SSEError'
    this.code = code
  }
}

export async function readSSE(body, { signal, onEvent, validateEvent = value => value } = {}) {
  if (!body || typeof body.getReader !== 'function') throw new SSEError()
  const reader = body.getReader()
  const decoder = new TextDecoder('utf-8', { fatal: true })
  let terminal = null, count = 0, total = 0, frameBytes = 0
  let lineBytes = 0, afterCR = false, pendingReset = false, ended = false
  let rejectAbort
  const aborted = new Promise((resolve, reject) => { rejectAbort = reject })
  const stop = () => {
    rejectAbort(new SSEError('request_aborted'))
    reader.cancel().catch(() => {})
  }
  const parser = createParser({
    maxBufferSize: MAX_EVENT_BYTES,
    onError() { throw new SSEError() },
    onEvent(value) {
      if (!EVENTS.has(value.event) || terminal || ++count > MAX_EVENTS) throw new SSEError()
      let data
      try { data = JSON.parse(value.data) } catch { throw new SSEError() }
      const verified = validateEvent({ event: value.event, data })
      if (value.event === 'done' || value.event === 'error') terminal = verified
      if (onEvent) onEvent(verified)
    },
  })

  function feed(bytes) {
    total += bytes.byteLength
    if (total > MAX_TOTAL_BYTES) throw new SSEError()
    // Count raw UTF-8 bytes at line boundaries; the library owns SSE semantics.
    let start = 0
    for (let index = 0; index < bytes.length; index++) {
      const byte = bytes[index]
      if (afterCR && byte === 10) {
        if (++frameBytes > MAX_EVENT_BYTES) throw new SSEError()
        if (pendingReset) frameBytes = 0
        pendingReset = false
        afterCR = false
        continue
      }
      if (pendingReset) frameBytes = 0
      pendingReset = false
      afterCR = false
      if (++frameBytes > MAX_EVENT_BYTES) throw new SSEError()
      if (byte === 10 || byte === 13) {
        if (lineBytes === 0) {
          if (byte === 13) pendingReset = true
          else frameBytes = 0
        }
        lineBytes = 0
        afterCR = byte === 13
        parser.feed(decoder.decode(bytes.subarray(start, index + 1), { stream: true }))
        start = index + 1
      } else lineBytes++
    }
    if (start < bytes.length) parser.feed(decoder.decode(bytes.subarray(start), { stream: true }))
  }

  try {
    if (signal) signal.addEventListener('abort', stop, { once: true })
    if (signal && signal.aborted) throw new SSEError('request_aborted')
    while (true) {
      const chunk = await Promise.race([reader.read(), aborted])
      if (signal && signal.aborted) throw new SSEError('request_aborted')
      if (chunk.done) break
      feed(chunk.value)
    }
    parser.feed(decoder.decode())
    if (afterCR) {
      parser.feed('\n')
      if (pendingReset) frameBytes = 0
    }
    if (!terminal || lineBytes > 0 || frameBytes > 0) throw new SSEError()
    ended = true
    return terminal
  } catch (error) {
    throw error instanceof SSEError ? error : new SSEError()
  } finally {
    if (signal) signal.removeEventListener('abort', stop)
    if (!ended) await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}

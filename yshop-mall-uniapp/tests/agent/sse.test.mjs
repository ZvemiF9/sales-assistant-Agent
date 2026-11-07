import assert from 'node:assert/strict'
import test from 'node:test'
import { readSSE, SSEError } from '../../utils/agent/sse.mjs'
import { createAgentClient } from '../../utils/agent/client.mjs'
import { createConversation } from '../../utils/agent/conversation.mjs'
import { action, card, chat, deferred, memoryStorage, product, source } from './fixtures.mjs'

const encode = value => new TextEncoder().encode(value)
const frame = (event, data, eol = '\n') => `event: ${event}${eol}data: ${JSON.stringify(data)}${eol}${eol}`
function response(parts, cancel = () => {}) {
  return new Response(new ReadableStream({
    start(controller) { for (const part of parts) controller.enqueue(typeof part === 'string' ? encode(part) : part); controller.close() },
    cancel,
  }), { headers: { 'Content-Type': 'text/event-stream' } })
}

test('SSE handles every byte split including multibyte UTF-8, CR, LF and CRLF', async () => {
  for (const ending of ['\n', '\r', '\r\n']) {
    const bytes = encode(': heartbeat' + ending + ending + frame('text', { text: '\u4f60\u597d' }, ending) + frame('done', chat(), ending))
    for (let split = 1; split < bytes.length; split++) {
      const seen = []
      const final = await readSSE(response([bytes.slice(0, split), bytes.slice(split)]).body, { onEvent: event => seen.push(event) })
      assert.equal(final.event, 'done')
      assert.deepEqual(final.data, chat())
      assert.equal(seen[0].data.text, '\u4f60\u597d')
    }
  }
})

test('SSE supports multiline JSON data and ignores comment-only frames', async () => {
  const final = await readSSE(response([': hello\n\n', 'event: done\ndata: {"value":\ndata: 1}\n\n']).body)
  assert.deepEqual(final.data, { value: 1 })
})

for (const [name, wire] of [
  ['malformed JSON', 'event: done\ndata: {bad}\n\n'],
  ['unknown event', frame('private', {})],
  ['missing done', frame('text', { text: 'partial' })],
  ['duplicate done', frame('done', {}) + frame('done', {})],
  ['after terminal', frame('done', {}) + frame('text', { text: 'late' })],
  ['missing event type', 'data: {}\n\n'],
  ['truncated UTF-8', new Uint8Array([0xe4, 0xbd])],
]) {
  test(`SSE rejects ${name}`, async () => {
    await assert.rejects(readSSE(response([wire]).body), SSEError)
  })
}

test('SSE enforces raw event bytes, cumulative bytes and business-event limits', async () => {
  await assert.rejects(readSSE(response(['data: ' + '\u4f60'.repeat(400000)]).body), SSEError)
  await assert.rejects(readSSE(response([Array.from({ length: 257 }, () => frame('status', { status: 'processing' })).join('')]).body), SSEError)
  const comments = ': ' + 'x'.repeat(60000) + '\n\n'
  await assert.rejects(readSSE(response(Array(150).fill(comments)).body), SSEError)
})

test('aborting a stalled reader cancels it and cannot accept a provisional reply', async () => {
  let cancelled = false
  const body = new ReadableStream({ start(controller) { controller.enqueue(encode(frame('status', { status: 'processing' }))) }, cancel() { cancelled = true } })
  const abort = new AbortController()
  const task = readSSE(body, { signal: abort.signal })
  abort.abort()
  await assert.rejects(task, error => error.code === 'request_aborted')
  assert.equal(cancelled, true)
})

test('stream client uses one authenticated POST, validates final JSON and leaves JSON mode compatible', async () => {
  const calls = [], seen = []
  const expected = chat({ products: [product()], sources: [source()] })
  const client = createAgentClient({ baseUrl: 'http://agent.test', getToken: () => 'test-token', fetchImpl: async (url, options) => {
    calls.push({ url, options })
    return response([frame('status', { status: 'processing' }), frame('products', { products: expected.products }), frame('done', expected)])
  } })
  assert.deepEqual(await client.chat({ message: 'query' }, { stream: true, onEvent: event => seen.push(event) }), expected)
  assert.equal(calls.length, 1)
  assert.equal(calls[0].url, 'http://agent.test/agent/chat/stream')
  assert.equal(calls[0].options.headers.Authorization, 'Bearer test-token')
  assert.equal(calls[0].options.headers.Accept, 'text/event-stream')
  assert.deepEqual(JSON.parse(calls[0].options.body), { message: 'query' })
  assert.equal(seen[0].event, 'status')
})

test('stream errors, changed credentials and leaked credentials are never accepted or automatically replayed', async () => {
  for (const kind of ['error', 'missing', 'credential', 'changed']) {
    let token = 'test-token', count = 0
    const seen = []
    const client = createAgentClient({ baseUrl: 'http://agent.test', getToken: () => token, fetchImpl: async () => {
      count++
      if (kind === 'changed') token = 'other-token'
      return response([kind === 'error' ? frame('error', { error: { code: 'model_timeout', message: 'private server exception' } })
        : kind === 'missing' ? frame('text', { text: 'partial' })
          : frame('done', chat({ answer: kind === 'credential' ? 'test-token' : 'reply' }))])
    } })
    await assert.rejects(client.chat({ message: 'query' }, { stream: true, onEvent: event => seen.push(event) }),
      error => !error.message.includes('private') && !error.message.includes('test-token'))
    assert.equal(count, 1)
    assert.ok(!JSON.stringify(seen).includes('test-token'))
  }
})

test('provisional stream data never enters cache, candidates or executable confirmation state', async () => {
  const gate = deferred(), storage = memoryStorage()
  let options
  const client = { chat: async (input, incoming) => { options = incoming; return gate.promise } }
  const conversation = createConversation({ client, storage })
  conversation.bindIdentity({ user_id: 11, nickname: 'test' })
  const running = conversation.send('query', { stream: true })
  options.onEvent({ event: 'status', data: { status: 'processing' } })
  options.onEvent({ event: 'confirmation', data: { pending_action: card() } })
  options.onEvent({ event: 'text', data: { text: 'provisional' } })
  assert.equal(conversation.state.pendingAction, null)
  assert.equal(conversation.state.canDecide, false)
  assert.equal(storage.values.size, 0)
  assert.equal(conversation.state.streamStatus, 'processing')
  gate.resolve(chat({ ...action(), model_rounds: 0 }))
  await running
  assert.ok(conversation.state.pendingAction)
  assert.equal(conversation.state.streamStatus, null)
})

test('new session invalidates late stream events and aborts the owned read', async () => {
  const gate = deferred()
  let options
  const conversation = createConversation({ client: { chat: async (input, incoming) => { options = incoming; return gate.promise } }, storage: memoryStorage() })
  conversation.bindIdentity({ user_id: 11, nickname: 'test' })
  const running = conversation.send('query', { stream: true })
  conversation.newSession()
  options.onEvent({ event: 'status', data: { status: 'processing' } })
  gate.resolve(chat())
  await running
  assert.equal(options.signal.aborted, true)
  assert.equal(conversation.state.streamStatus, null)
  assert.deepEqual(conversation.state.messages, [])
})

test('stream keeps fixed public model and authentication errors, without upstream text', async () => {
  for (const code of ['model_timeout', 'login_required']) {
    const client = createAgentClient({ baseUrl: 'http://agent.test', getToken: () => 'test-token',
      fetchImpl: async () => response([frame('error', { error: { code, message: 'private' } })]) })
    await assert.rejects(client.chat({ message: 'query' }, { stream: true }), error => error.code === code && !error.message.includes('private'))
  }
})

test('identity changes between stream events clear identity rather than becoming transport errors', async () => {
  let token = 'test-token'
  const client = createAgentClient({ baseUrl: 'http://agent.test', getToken: () => token,
    fetchImpl: async () => response([frame('status', { status: 'processing' }), frame('done', chat())]) })
  await assert.rejects(client.chat({ message: 'query' }, { stream: true, onEvent() { token = 'other-token' } }),
    error => error.code === 'login_required' && error.status === 401)
})

test('raw frame bound includes both bytes of CRLF', async () => {
  const max = 1024 * 1024
  const prefix = 'event: done\r\ndata: {"text":"'
  const suffix = '"}\r\n\r\n'
  const wire = prefix + 'x'.repeat(max + 1 - encode(prefix + suffix).length) + suffix
  assert.equal(encode(wire).length, max + 1)
  await assert.rejects(readSSE(response([wire]).body), SSEError)
})

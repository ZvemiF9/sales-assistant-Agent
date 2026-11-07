import assert from 'node:assert/strict'
import test from 'node:test'
import { AgentClientError, createAgentClient } from '../../utils/agent/client.mjs'
import { action, actionId, chat, deferred, history, jsonResponse, product, replacementId, requestId, result, sessionId, source } from './fixtures.mjs'

test('chat accepts Agent JSON and sends the current Bearer header', async () => {
  const observed = []
  const response = {
    session_id: 'a'.repeat(32), answer: '请选择商品。', status: 'needs_input',
    products: [], sources: [], notice: null, tool_calls: [],
    model_rounds: 1, usage: { input_tokens: null, output_tokens: null },
    pending_action: null, action_result: null,
  }
  const client = createAgentClient({
    baseUrl: 'http://127.0.0.1:8018',
    getToken: () => 'test-only-token',
    fetchImpl: async (url, options) => {
      observed.push({ url, options })
      return new Response(JSON.stringify(response), {
        status: 200, headers: { 'Content-Type': 'application/json' },
      })
    },
  })
  assert.equal((await client.chat({ message: '查商品' })).status, 'needs_input')
  assert.equal(observed.length, 1)
  assert.equal(observed[0].options.headers.Authorization, 'Bearer test-only-token')
})

test('each request reads the current token and uses only the independent Agent origin', async () => {
  const observed = []
  let token = 'first-test-token'
  const client = createAgentClient({
    baseUrl: 'http://127.0.0.1:8018/',
    getToken: () => token,
    fetchImpl: async (url, options) => {
      observed.push({ url, options })
      return jsonResponse({ user_id: 11, nickname: '测试用户' })
    },
  })
  await client.me()
  token = 'second-test-token'
  await client.me()
  assert.deepEqual(observed.map(item => item.url), ['http://127.0.0.1:8018/agent/me', 'http://127.0.0.1:8018/agent/me'])
  assert.deepEqual(observed.map(item => item.options.headers.Authorization), ['Bearer first-test-token', 'Bearer second-test-token'])
  assert.equal(observed[0].options.credentials, 'omit')
  assert.equal(observed[0].options.cache, 'no-store')
  assert.equal(observed[0].options.redirect, 'error')
  assert.equal(observed[0].options.headers.Accept, 'application/json')
})

test('chat sends only normalized message and optional valid session ID as JSON', async () => {
  let captured
  const abort = new AbortController()
  const client = createAgentClient({
    baseUrl: 'http://127.0.0.1:8018',
    getToken: () => 'test-only-token',
    fetchImpl: async (url, options) => { captured = { url, options }; return jsonResponse(chat()) },
  })
  await client.chat({ message: '  查商品  ', session_id: sessionId }, { signal: abort.signal })
  assert.equal(captured.url, 'http://127.0.0.1:8018/agent/chat')
  assert.equal(captured.options.method, 'POST')
  assert.equal(captured.options.headers['Content-Type'], 'application/json')
  assert.deepEqual(JSON.parse(captured.options.body), { message: '查商品', session_id: sessionId })
  assert.equal(captured.options.signal, abort.signal)
})

test('message validation matches trimmed 1 through 1000 Unicode characters', async () => {
  let calls = 0
  const client = createAgentClient({
    baseUrl: 'http://127.0.0.1:8018',
    getToken: () => 'test-only-token',
    fetchImpl: async () => { calls += 1; return jsonResponse(chat()) },
  })
  await client.chat({ message: '杯'.repeat(1000) })
  await client.chat({ message: '\u{1f600}'.repeat(1000) })
  for (const message of ['', '  ', 12, null, undefined, '杯'.repeat(1001), '\u{1f600}'.repeat(1001)]) {
    await assert.rejects(client.chat({ message }), { code: 'invalid_request' })
  }
  assert.equal(calls, 2)
})

test('invalid session, extra request fields and accidental current credential are rejected before fetching', async () => {
  let calls = 0
  const client = createAgentClient({
    baseUrl: 'http://127.0.0.1:8018',
    getToken: () => 'test-only-token',
    fetchImpl: async () => { calls += 1; return jsonResponse(chat()) },
  })
  for (const input of [
    { message: '查商品', session_id: 12 },
    { message: '查商品', session_id: 'A'.repeat(32) },
    { message: '查商品', session_id: '../action' },
    { message: '查商品', product_id: 101 },
    { message: '我的 test-only-token' },
  ]) await assert.rejects(client.chat(input), { code: 'invalid_request' })
  assert.equal(calls, 0)
})

test('missing, non-string and pre-prefixed credentials are login errors without a network call', async () => {
  for (const token of [undefined, null, '', ' ', 12, 'Bearer test-only-token', 'secret\nheader']) {
    let calls = 0
    const client = createAgentClient({
      baseUrl: 'http://127.0.0.1:8018', getToken: () => token,
      fetchImpl: async () => { calls += 1; return jsonResponse(chat()) },
    })
    await assert.rejects(client.me(), { code: 'login_required', status: 401 })
    assert.equal(calls, 0)
  }
})

test('actions and history use strict ID paths; a decision body contains only decision', async () => {
  const observed = []
  const client = createAgentClient({
    baseUrl: 'http://127.0.0.1:8018', getToken: () => 'test-only-token',
    fetchImpl: async (url, options) => {
      observed.push({ url, options })
      return jsonResponse(url.endsWith('/history') ? history() : action())
    },
  })
  await client.getAction(actionId)
  await client.getHistory(actionId)
  await client.decide(actionId, 'confirm')
  await client.decide(actionId, 'cancel')
  assert.deepEqual(observed.map(item => item.url), [
    `http://127.0.0.1:8018/agent/actions/${actionId}`,
    `http://127.0.0.1:8018/agent/actions/${actionId}/history`,
    `http://127.0.0.1:8018/agent/actions/${actionId}/decision`,
    `http://127.0.0.1:8018/agent/actions/${actionId}/decision`,
  ])
  assert.deepEqual(JSON.parse(observed[2].options.body), { decision: 'confirm' })
  assert.deepEqual(JSON.parse(observed[3].options.body), { decision: 'cancel' })
  const before = observed.length
  for (const id of [101, '../decision', 'b'.repeat(31), Number.MAX_SAFE_INTEGER + 1]) {
    await assert.rejects(client.getAction(id), { code: 'invalid_request' })
    await assert.rejects(client.getHistory(id), { code: 'invalid_request' })
    await assert.rejects(client.decide(id, 'confirm'), { code: 'invalid_request' })
  }
  for (const decision of ['yes', { decision: 'confirm' }, null]) {
    await assert.rejects(client.decide(actionId, decision), { code: 'invalid_request' })
  }
  assert.equal(observed.length, before)
})

test('network failure attempts exactly once and never exposes original error or credential', async () => {
  let calls = 0
  const secret = 'test-only-token'
  const original = new Error(`fetch failed with ${secret} private server body`)
  const client = createAgentClient({
    baseUrl: 'http://127.0.0.1:8018', getToken: () => secret,
    fetchImpl: async () => { calls += 1; throw original },
  })
  await assert.rejects(client.decide(actionId, 'confirm'), error => {
    assert.ok(error instanceof AgentClientError)
    assert.equal(error.code, 'network_error')
    assert.equal(error.status, 0)
    assert.equal(error.cause, undefined)
    assert.equal(error.originalError, undefined)
    assert.equal(error.payload, undefined)
    assert.ok(!JSON.stringify(error).includes(secret))
    assert.ok(!String(error).includes(secret))
    return true
  })
  assert.equal(calls, 1)
})

test('401 and public server errors use fixed public messages and a validated request ID', async () => {
  for (const [status, code] of [[401, 'login_required'], [409, 'session_conflict'], [502, 'model_unavailable']]) {
    let calls = 0
    const client = createAgentClient({
      baseUrl: 'http://127.0.0.1:8018', getToken: () => 'test-only-token',
      fetchImpl: async () => {
        calls += 1
        return jsonResponse({ error: { code, message: 'Bearer test-only-token private' }, debug: 'private' }, status, { 'X-Request-ID': requestId })
      },
    })
    await assert.rejects(client.chat({ message: '查商品' }), error => {
      assert.equal(error.code, code)
      assert.equal(error.status, status)
      assert.equal(error.requestId, requestId)
      assert.ok(!JSON.stringify(error).includes('test-only-token'))
      assert.ok(!error.message.includes('private'))
      return true
    })
    assert.equal(calls, 1)
  }
})

test('unrecognized server payload and malformed request IDs are not reflected', async () => {
  const client = createAgentClient({
    baseUrl: 'http://127.0.0.1:8018', getToken: () => 'test-only-token',
    fetchImpl: async () => jsonResponse({ error: { code: 'Bearer-test-only-token', message: 'secret' } }, 503, { 'X-Request-ID': 'test-only-token' }),
  })
  await assert.rejects(client.me(), error => {
    assert.equal(error.code, 'agent_unavailable')
    assert.equal(error.requestId, undefined)
    assert.ok(!JSON.stringify(error).includes('test-only-token'))
    return true
  })
})

test('non-JSON, malformed, oversized and shop-wrapped success are invalid Agent responses', async () => {
  for (const response of [
    new Response('<html>secret</html>', { status: 200, headers: { 'Content-Type': 'text/html' } }),
    new Response('{private', { status: 200, headers: { 'Content-Type': 'application/json' } }),
    jsonResponse({ status: 200, data: chat() }),
    jsonResponse(chat({ answer: 'x'.repeat(1024 * 1024 + 1) })),
  ]) {
    const client = createAgentClient({
      baseUrl: 'http://127.0.0.1:8018', getToken: () => 'test-only-token', fetchImpl: async () => response,
    })
    await assert.rejects(client.chat({ message: '查商品' }), { code: 'agent_invalid_response' })
  }
})

test('unsafe product, user and cart IDs, non-finite values and inconsistent action totals are rejected', async () => {
  for (const payload of [
    chat({ products: [product({ product_id: Number.MAX_SAFE_INTEGER + 1 })] }),
    chat({ products: [product({ skus: [{ name: '红色', unique_id: 'red', price: 'Infinity', stock: 1, image: '' }] })] }),
    chat({ sources: [source({ score: null })] }),
    action({ pending_action: { ...action().pending_action, total_price: '1.00' } }),
    action({ pending_action: { ...action().pending_action, quantity: 10000 } }),
    action({ pending_action: null, status: 'succeeded', action_result: result('succeeded', { cart_id: Number.MAX_SAFE_INTEGER + 1 }) }),
  ]) {
    const client = createAgentClient({
      baseUrl: 'http://127.0.0.1:8018', getToken: () => 'test-only-token',
      fetchImpl: async () => jsonResponse(payload),
    })
    const request = 'products' in payload ? client.chat({ message: '查商品' }) : client.getAction(actionId)
    await assert.rejects(request, { code: 'agent_invalid_response' })
  }
  const client = createAgentClient({
    baseUrl: 'http://127.0.0.1:8018', getToken: () => 'test-only-token',
    fetchImpl: async () => jsonResponse({ user_id: Number.MAX_SAFE_INTEGER + 1, nickname: '' }),
  })
  await assert.rejects(client.me(), { code: 'agent_invalid_response' })
})

test('complete public products, sources and action history are accepted without private extras', async () => {
  const client = createAgentClient({
    baseUrl: 'http://127.0.0.1:8018', getToken: () => 'test-only-token',
    fetchImpl: async url => jsonResponse(url.endsWith('/history') ? history() : chat({
      products: [product({ private_tool_args: 'hidden' })],
      sources: [source()], notice: '仅为演示规则。', raw_response: 'hidden', reasoning: 'hidden',
    })),
  })
  const response = await client.chat({ message: '退货规则' })
  assert.equal(response.products[0].skus[0].price, '69.90')
  assert.equal(response.sources[0].text, source().text)
  assert.equal(response.raw_response, undefined)
  assert.equal(response.reasoning, undefined)
  assert.equal(response.products[0].private_tool_args, undefined)
  assert.equal((await client.getHistory(actionId)).events[0].kind, 'prepared')
})

test('current credentials are rejected if returned in public fields and never embedded in an image URL', async () => {
  for (const payload of [
    chat({ answer: 'token test-only-token' }),
    chat({ products: [product({ image: 'https://example.com/image?token=test-only-token' })] }),
  ]) {
    const client = createAgentClient({
      baseUrl: 'http://127.0.0.1:8018', getToken: () => 'test-only-token',
      fetchImpl: async () => jsonResponse(payload),
    })
    await assert.rejects(client.chat({ message: '查商品' }), { code: 'agent_invalid_response' })
  }
})

test('abort is public and distinct from network or authentication failure', async () => {
  const abort = new AbortController()
  abort.abort()
  const client = createAgentClient({
    baseUrl: 'http://127.0.0.1:8018', getToken: () => 'test-only-token',
    fetchImpl: async () => { throw new DOMException('private abort details', 'AbortError') },
  })
  await assert.rejects(client.chat({ message: '查商品' }, { signal: abort.signal }), { code: 'request_aborted', status: 0 })
})

test('unsafe API URLs are rejected without reflecting their embedded values', () => {
  for (const baseUrl of ['http://user:secret@example.com', 'http://example.com?token=secret', 'javascript:secret', 'http://example.com/#secret']) {
    assert.throws(() => createAgentClient({ baseUrl, getToken: () => 'test-only-token', fetchImpl: async () => {} }), error => {
      assert.equal(error.code, 'invalid_configuration')
      assert.ok(!String(error).includes('secret'))
      return true
    })
  }
})

test('a credential change during an in-flight request invalidates the old response', async () => {
  const gate = deferred()
  let token = 'old-test-token'
  const client = createAgentClient({
    baseUrl: 'http://127.0.0.1:8018', getToken: () => token, fetchImpl: async () => gate.promise,
  })
  const response = client.chat({ message: '查商品' })
  token = 'new-test-token'
  gate.resolve(jsonResponse(chat()))
  await assert.rejects(response, { code: 'login_required', status: 401 })
})

test('action reads and histories must match their path; decisions allow only a new pending replacement ID', async () => {
  for (const [method, payload] of [
    ['getAction', action({ pending_action: { ...action().pending_action, action_id: replacementId } })],
    ['getHistory', history({ action_id: replacementId })],
    ['decide', action({ status: 'succeeded', pending_action: null, action_result: result('succeeded', { action_id: replacementId }) })],
    ['getAction', action({ status: 'needs_input', pending_action: null })],
    ['decide', action({ status: 'needs_input', pending_action: null })],
  ]) {
    const client = createAgentClient({
      baseUrl: 'http://127.0.0.1:8018', getToken: () => 'test-only-token', fetchImpl: async () => jsonResponse(payload),
    })
    await assert.rejects(method === 'decide' ? client.decide(actionId, 'confirm') : client[method](actionId), { code: 'agent_invalid_response' })
  }
  const client = createAgentClient({
    baseUrl: 'http://127.0.0.1:8018', getToken: () => 'test-only-token',
    fetchImpl: async () => jsonResponse(action({ pending_action: { ...action().pending_action, action_id: replacementId } })),
  })
  assert.equal((await client.decide(actionId, 'confirm')).pending_action.action_id, replacementId)
})

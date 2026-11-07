import assert from 'node:assert/strict'
import test from 'node:test'
import { AgentClientError } from '../../utils/agent/client.mjs'
import { createConversation } from '../../utils/agent/conversation.mjs'
import { action, actionId, card, chat, deferred, fakeClient, memoryStorage, product, replacementId, result, sessionId } from './fixtures.mjs'

const now = () => 1893456000000

function setup(overrides = {}) {
  const client = fakeClient(overrides)
  const storage = memoryStorage()
  const conversation = createConversation({ client, storage, now })
  conversation.bindIdentity({ user_id: 11, nickname: '测试用户' })
  return { conversation, client, storage }
}

test('identity verification binds only the actual Agent identity, not cached shop user data', async () => {
  const client = fakeClient()
  const conversation = createConversation({ client, storage: memoryStorage(), now })
  assert.equal(conversation.state.identity, null)
  assert.equal(conversation.state.authStatus, 'unverified')
  await conversation.verifyIdentity()
  assert.equal(conversation.state.identity.user_id, 11)
  assert.equal(conversation.state.authStatus, 'authenticated')
})

test('an unverified identity cannot send or prepare and unsafe identities are rejected', async () => {
  const client = fakeClient()
  const conversation = createConversation({ client, storage: memoryStorage(), now })
  await conversation.send('查商品')
  assert.equal(client.observed.length, 0)
  assert.equal(conversation.state.error.code, 'login_required')
  assert.throws(() => conversation.bindIdentity({ user_id: Number.MAX_SAFE_INTEGER + 1, nickname: '' }), { code: 'agent_invalid_response' })
})

test('sending continues with the current session and stores only public text', async () => {
  const { conversation, client, storage } = setup({
    chat: async (input, options) => {
      client.observed.push({ method: 'chat', input, options })
      return chat({ products: [product()], tool_calls: [{ name: 'search_products', status: 'succeeded', elapsed_ms: 1 }], reasoning: 'hidden' })
    },
  })
  await conversation.send('查商品')
  await conversation.send('只看红色')
  assert.deepEqual(client.observed.map(item => item.input), [{ message: '查商品' }, { message: '只看红色', session_id: sessionId }])
  assert.equal(conversation.state.sessionId, sessionId)
  assert.equal(conversation.state.messages.length, 4)
  const cached = [...storage.values.values()].join('')
  assert.ok(!cached.includes('tool_calls'))
  assert.ok(!cached.includes('reasoning'))
  assert.ok(!cached.includes('Authorization'))
})

test('only the latest candidate revision can prepare an explicit backend-compatible message', async () => {
  const { conversation, client } = setup({
    chat: async (input, options) => {
      client.observed.push({ method: 'chat', input, options })
      return input.message.startsWith('第') ? chat({ ...action(), model_rounds: 0 }) : chat({ products: [product()] })
    },
  })
  await conversation.send('查商品')
  const revision = conversation.state.candidateRevision
  await conversation.prepare({ productId: 101, uniqueId: 'cup-red', quantity: 2, revision })
  assert.equal(client.observed[1].input.message, '第1个 红色规格，加入2件购物车')
  assert.deepEqual(Object.keys(client.observed[1].input).sort(), ['message', 'session_id'])
  assert.equal(conversation.state.pendingAction.action_id, actionId)
  assert.equal(conversation.state.canDecide, true)
  assert.equal(client.observed.some(item => item.method === 'decide'), false)
})

test('old candidates become unavailable immediately while a new query is in flight', async () => {
  const gate = deferred()
  let calls = 0
  const { conversation, client } = setup({ chat: async () => { calls += 1; return calls === 1 ? chat({ products: [product()] }) : gate.promise } })
  await conversation.send('查商品')
  const revision = conversation.state.candidateRevision
  const sending = conversation.send('换一种商品')
  assert.deepEqual(conversation.state.candidates, [])
  await conversation.prepare({ productId: 101, uniqueId: 'cup-red', quantity: 2, revision })
  assert.equal(calls, 2)
  assert.equal(client.observed.some(item => item.method === 'decide'), false)
  gate.resolve(chat({ products: [product({ product_id: 202 })] }))
  await sending
  await conversation.prepare({ productId: 101, uniqueId: 'cup-red', quantity: 2, revision })
  assert.equal(calls, 2)
  assert.equal(conversation.state.error.code, 'stale_candidate')
})

test('quantity, safe numeric product ID and exact available SKU are checked before preparation', async () => {
  const { conversation, client } = setup()
  await conversation.send('查商品')
  const revision = conversation.state.candidateRevision
  const count = client.observed.length
  for (const quantity of [0, 10000, 1.5, NaN, Infinity, '2', true]) {
    await conversation.prepare({ productId: 101, uniqueId: 'cup-red', quantity, revision })
  }
  for (const productId of [0, '101', NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
    await conversation.prepare({ productId, uniqueId: 'cup-red', quantity: 2, revision })
  }
  await conversation.prepare({ productId: 101, uniqueId: 'unknown-sku', quantity: 2, revision })
  assert.equal(client.observed.length, count)
})

test('SKU labels that are ambiguous or interpreted as instructions cannot prepare a different action', async () => {
  for (const skus of [
    [{ name: '标准版', unique_id: 'a', price: '1', stock: 5, image: '' }, { name: '版', unique_id: 'b', price: '2', stock: 5, image: '' }],
    [{ name: '不要', unique_id: 'a', price: '1', stock: 5, image: '' }],
    [{ name: '123', unique_id: 'a', price: '1', stock: 5, image: '' }],
    [{ name: '红色"', unique_id: 'a', price: '1', stock: 5, image: '' }],
  ]) {
    const { conversation, client } = setup({ chat: async () => chat({ products: [product({ skus })] }) })
    await conversation.send('查商品')
    await conversation.prepare({ productId: 101, uniqueId: 'a', quantity: 2, revision: conversation.state.candidateRevision })
    assert.equal(client.observed.length, 0)
    assert.equal(conversation.state.error.code, 'sku_selection')
  }
})

test('pending confirmation snapshots are cloned, frozen and cannot change under the same action ID', async () => {
  const initial = card()
  const { conversation } = setup({ chat: async () => chat({ ...action({ pending_action: initial }), model_rounds: 0 }) })
  await conversation.send('准备加购')
  initial.unit_price = '1.00'
  assert.equal(conversation.state.pendingAction.unit_price, '69.90')
  assert.ok(Object.isFrozen(conversation.state.pendingAction))
  assert.throws(() => { conversation.state.pendingAction.quantity = 8 }, TypeError)
})

test('new session invalidates late chat responses and aborts the in-flight request', async () => {
  const gate = deferred()
  let signal
  const { conversation } = setup({ chat: async (input, options) => { signal = options.signal; return gate.promise } })
  const sending = conversation.send('查商品')
  conversation.newSession()
  assert.equal(signal.aborted, true)
  gate.resolve(chat({ products: [product()] }))
  await sending
  assert.equal(conversation.state.sessionId, null)
  assert.deepEqual(conversation.state.messages, [])
  assert.deepEqual(conversation.state.candidates, [])
  assert.equal(conversation.state.busy, false)
})

test('account switching clears visible state and ignores late responses from the previous account', async () => {
  const gate = deferred()
  const { conversation, storage } = setup({ chat: async () => gate.promise })
  const sending = conversation.send('甲账户的消息')
  conversation.bindIdentity({ user_id: 22, nickname: '乙账户' })
  gate.resolve(chat({ products: [product()] }))
  await sending
  assert.equal(conversation.state.identity.user_id, 22)
  assert.deepEqual(conversation.state.messages, [])
  assert.deepEqual(conversation.state.candidates, [])
  assert.ok(![...storage.values.values()].join('').includes('甲账户'))
})

test('logout and expired authentication remove the account cache and invalidate pending asynchronous work', async () => {
  const { conversation, storage, client } = setup()
  await conversation.send('查商品')
  assert.equal(storage.values.size, 1)
  client.chat = async () => { throw new AgentClientError('login_required', 401) }
  await conversation.send('查规则')
  assert.equal(conversation.state.authStatus, 'login_required')
  assert.equal(conversation.state.identity, null)
  assert.deepEqual(conversation.state.messages, [])
  assert.equal(storage.values.size, 0)
  conversation.bindIdentity({ user_id: 11, nickname: '测试用户' })
  conversation.clearIdentity()
  assert.equal(conversation.state.identity, null)
  assert.equal(conversation.state.authStatus, 'login_required')
})

test('late identity verification cannot rebind a logged-out or replaced account', async () => {
  const gate = deferred()
  const conversation = createConversation({ client: fakeClient({ me: async () => gate.promise }), storage: memoryStorage(), now })
  const checking = conversation.verifyIdentity()
  conversation.clearIdentity()
  gate.resolve({ user_id: 11, nickname: '旧账户' })
  await checking
  assert.equal(conversation.state.identity, null)
  assert.equal(conversation.state.authStatus, 'login_required')
})

test('double-send and duplicate decision clicks each cause exactly one attempt', async () => {
  const chatGate = deferred()
  const decisionGate = deferred()
  let chats = 0
  let decisions = 0
  const { conversation } = setup({
    chat: async () => { chats += 1; return chatGate.promise },
    decide: async () => { decisions += 1; return decisionGate.promise },
  })
  const sending = conversation.send('准备加购')
  await conversation.send('重复发送')
  assert.equal(chats, 1)
  chatGate.resolve(chat({ ...action(), model_rounds: 0 }))
  await sending
  const deciding = conversation.decide(actionId, 'confirm')
  await conversation.decide(actionId, 'confirm')
  await conversation.decide(actionId, 'cancel')
  assert.equal(decisions, 1)
  assert.equal(conversation.state.canDecide, false)
  decisionGate.resolve(action({ status: 'succeeded', pending_action: null, action_result: result() }))
  await deciding
  await conversation.decide(actionId, 'confirm')
  assert.equal(decisions, 1)
  assert.equal(conversation.state.actionResult.status, 'succeeded')
})

test('a replacement ID requires a separate explicit decision and never mutates the original snapshot', async () => {
  const original = card()
  const fresh = card({ action_id: replacementId, unit_price: '79.90', total_price: '159.80' })
  let decisions = 0
  const { conversation } = setup({
    chat: async () => chat({ ...action({ pending_action: original }), model_rounds: 0 }),
    decide: async () => { decisions += 1; return action({ pending_action: fresh }) },
  })
  await conversation.send('准备加购')
  const snapshot = conversation.state.pendingAction
  await conversation.decide(actionId, 'confirm')
  assert.equal(decisions, 1)
  assert.equal(snapshot.unit_price, '69.90')
  assert.equal(conversation.state.pendingAction.action_id, replacementId)
  assert.equal(conversation.state.canDecide, true)
  await conversation.decide(actionId, 'confirm')
  assert.equal(decisions, 1)
})

test('a changed snapshot under the same ID remains read-only after a decision', async () => {
  let decisions = 0
  const { conversation } = setup({
    chat: async () => chat({ ...action(), model_rounds: 0 }),
    decide: async () => { decisions += 1; return action({ pending_action: card({ unit_price: '79.90', total_price: '159.80' }) }) },
  })
  await conversation.send('准备加购')
  await conversation.decide(actionId, 'confirm')
  assert.equal(conversation.state.canDecide, false)
  assert.equal(conversation.state.recoveryOnly, true)
  assert.equal(conversation.state.pendingAction.unit_price, '69.90')
  assert.equal(conversation.state.error.code, 'agent_invalid_response')
  await conversation.decide(actionId, 'confirm')
  assert.equal(decisions, 1)
})

test('uncertain decisions retain the action reference and allow GET recovery only, even if GET says pending', async () => {
  let decisions = 0
  let reads = 0
  const { conversation, storage } = setup({
    chat: async () => chat({ ...action(), model_rounds: 0 }),
    decide: async () => { decisions += 1; throw new AgentClientError('network_error', 0) },
    getAction: async () => { reads += 1; return action() },
  })
  await conversation.send('准备加购')
  await conversation.decide(actionId, 'confirm')
  assert.equal(conversation.state.actionId, actionId)
  assert.equal(conversation.state.recoveryOnly, true)
  assert.equal(conversation.state.canDecide, false)
  await conversation.recoverAction()
  await conversation.decide(actionId, 'confirm')
  await conversation.decide(actionId, 'cancel')
  assert.equal(decisions, 1)
  assert.equal(reads, 1)
  assert.ok([...storage.values.values()].join('').includes(actionId))
  assert.equal(conversation.state.canDecide, false)
})

test('executing and unknown results are never replayed as decisions or new chat preparation', async () => {
  for (const status of ['executing', 'unknown']) {
    let decisions = 0
    let chats = 0
    const { conversation } = setup({
      chat: async () => { chats += 1; return chat({ ...action(), model_rounds: 0 }) },
      decide: async () => { decisions += 1; return action({ status, pending_action: null, action_result: result(status) }) },
      getAction: async () => action({ status, pending_action: null, action_result: result(status) }),
    })
    await conversation.send('准备加购')
    await conversation.decide(actionId, 'confirm')
    await conversation.decide(actionId, 'confirm')
    await conversation.send('再加一件')
    await conversation.recoverAction()
    assert.equal(decisions, 1)
    assert.equal(chats, 1)
    assert.equal(conversation.state.canDecide, false)
    assert.equal(conversation.state.actionResult.status, status)
  }
})

test('cached pending actions are read-only until explicit GET refresh; cached products cannot prepare', async () => {
  const { conversation, client, storage } = setup({
    chat: async () => chat({ ...action(), model_rounds: 0 }),
  })
  await conversation.send('准备加购')
  conversation.dispose()
  const restored = createConversation({ client, storage, now })
  restored.bindIdentity({ user_id: 11, nickname: '测试用户' })
  assert.equal(restored.state.actionId, actionId)
  assert.equal(restored.state.pendingAction, null)
  assert.equal(restored.state.canDecide, false)
  assert.deepEqual(restored.state.candidates, [])
  assert.equal(client.observed.some(item => item.method === 'getAction'), false)
  await restored.decide(actionId, 'confirm')
  assert.equal(client.observed.some(item => item.method === 'decide'), false)
  await restored.recoverAction()
  assert.equal(restored.state.pendingAction.action_id, actionId)
  assert.equal(restored.state.canDecide, true)
})

test('uncertain decision lock survives page disposal and cache restoration', async () => {
  let decisions = 0
  const { conversation, client, storage } = setup({
    chat: async () => chat({ ...action(), model_rounds: 0 }),
    decide: async () => { decisions += 1; throw new AgentClientError('network_error', 0) },
  })
  await conversation.send('准备加购')
  await conversation.decide(actionId, 'confirm')
  conversation.dispose()
  const restored = createConversation({ client, storage, now })
  restored.bindIdentity({ user_id: 11, nickname: '测试用户' })
  await restored.recoverAction()
  await restored.decide(actionId, 'confirm')
  assert.equal(restored.state.recoveryOnly, true)
  assert.equal(restored.state.canDecide, false)
  assert.equal(decisions, 1)
})

test('an expired snapshot cannot enable confirmation', async () => {
  let decisions = 0
  const { conversation } = setup({
    chat: async () => chat({ ...action({ pending_action: card({ expires_at: 1893455999 }) }), model_rounds: 0 }),
    decide: async () => { decisions += 1; return action() },
  })
  await conversation.send('准备加购')
  await conversation.decide(actionId, 'confirm')
  assert.equal(conversation.state.canDecide, false)
  assert.equal(conversation.state.error.code, 'action_expired')
  assert.equal(decisions, 0)
})

test('malformed, oversized, cross-account and credential-containing caches are discarded', () => {
  const key = 'yshop-agent:conversation:v1:user:11'
  const good = { version: 1, user_id: 11, session_id: sessionId, messages: [{ role: 'assistant', content: '你好' }], action_id: null, action_locked: false }
  for (const raw of [
    '{broken',
    'x'.repeat(65537),
    JSON.stringify({ ...good, user_id: 22 }),
    JSON.stringify({ ...good, messages: [{ role: 'system', content: 'hidden' }] }),
    JSON.stringify({ ...good, token: 'test-only-token' }),
    JSON.stringify({ ...good, messages: [{ role: 'user', content: 'Bearer test-only-token' }] }),
    JSON.stringify({ ...good, action_id: '../decision' }),
  ]) {
    const storage = memoryStorage()
    storage.setItem(key, raw)
    const conversation = createConversation({ client: fakeClient(), storage, now })
    conversation.bindIdentity({ user_id: 11, nickname: '测试用户' })
    assert.deepEqual(conversation.state.messages, [])
    assert.equal(conversation.state.actionId, null)
    assert.equal(storage.getItem(key), null)
  }
})

test('chat cache is bounded to twelve messages and at most 64KiB UTF-8', async () => {
  const { conversation, storage } = setup({ chat: async () => chat({ answer: '答'.repeat(12000) }) })
  for (let index = 0; index < 18; index += 1) await conversation.send(`查询${index}`)
  assert.ok(conversation.state.messages.length <= 12)
  for (const cached of storage.values.values()) {
    assert.ok(Buffer.byteLength(cached, 'utf8') <= 65536)
    const payload = JSON.parse(cached)
    assert.ok(payload.messages.length <= 12)
    assert.ok(!('token' in payload))
    assert.ok(!('products' in payload))
    assert.ok(!('pending_action' in payload))
  }
})

test('storage failure does not cause network retries or lose the verified visible response', async () => {
  let calls = 0
  const storage = { getItem: () => { throw new Error('private') }, setItem: () => { throw new Error('private') }, removeItem: () => { throw new Error('private') } }
  const conversation = createConversation({ client: fakeClient({ chat: async () => { calls += 1; return chat() } }), storage, now })
  conversation.bindIdentity({ user_id: 11, nickname: '测试用户' })
  await conversation.send('查商品')
  assert.equal(calls, 1)
  assert.equal(conversation.state.messages.at(-1).content, '请选择商品。')
})

test('dispose invalidates late decision success and leaves no visible identity or result', async () => {
  const gate = deferred()
  const { conversation } = setup({
    chat: async () => chat({ ...action(), model_rounds: 0 }),
    decide: async () => gate.promise,
  })
  await conversation.send('准备加购')
  const deciding = conversation.decide(actionId, 'confirm')
  conversation.dispose()
  gate.resolve(action({ status: 'succeeded', pending_action: null, action_result: result() }))
  await deciding
  assert.equal(conversation.state.identity, null)
  assert.equal(conversation.state.actionResult, null)
  assert.deepEqual(conversation.state.messages, [])
})

test('SKU names that erase the generated quantity grammar are rejected before sending', async () => {
  let calls = 0
  const { conversation } = setup({
    chat: async () => {
      calls += 1
      return chat({ products: [product({ skus: [{ name: '规格', unique_id: 'special', price: '1.00', stock: 9, image: '' }] })] })
    },
  })
  await conversation.send('查商品')
  await conversation.prepare({ productId: 101, uniqueId: 'special', quantity: 2, revision: conversation.state.candidateRevision })
  assert.equal(calls, 1)
  assert.equal(conversation.state.error.code, 'sku_selection')
})

test('an unrelated decision outcome cannot overwrite the owned action reference', async () => {
  const { conversation } = setup({
    chat: async () => chat({ ...action(), model_rounds: 0 }),
    decide: async () => action({ status: 'succeeded', pending_action: null, action_result: result('succeeded', { action_id: replacementId }) }),
  })
  await conversation.send('准备加购')
  await conversation.decide(actionId, 'confirm')
  assert.equal(conversation.state.actionId, actionId)
  assert.equal(conversation.state.actionResult, null)
  assert.equal(conversation.state.canDecide, false)
  assert.equal(conversation.state.error.code, 'agent_invalid_response')
})

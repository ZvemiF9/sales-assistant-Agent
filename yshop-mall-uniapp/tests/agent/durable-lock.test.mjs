import assert from 'node:assert/strict'
import test from 'node:test'
import { AgentClientError } from '../../utils/agent/client.mjs'
import { createConversation } from '../../utils/agent/conversation.mjs'
import { action, actionId, chat, fakeClient, memoryStorage, result, sessionId } from './fixtures.mjs'

const now = () => 1893456000000
const identity = { user_id: 11, nickname: 'Test user' }
const key = 'yshop-agent:conversation:v1:user:11'
const privateFailure = 'private storage exception with private-token'

function seedPending(storage) {
  storage.setItem(key, JSON.stringify({
    version: 1, user_id: 11, session_id: sessionId,
    messages: [{ role: 'assistant', content: 'Pending action' }],
    action_id: actionId, action_locked: false,
  }))
}

function bind(client, storage) {
  const conversation = createConversation({ client, storage, now })
  conversation.bindIdentity(identity)
  return conversation
}

for (const decision of ['confirm', 'cancel']) {
  test(`failed durable ${decision} locking cannot replay an old unlocked cache after restoration`, async () => {
    const storage = memoryStorage()
    seedPending(storage)
    storage.setItem = () => { throw new Error(privateFailure) }
    let attempts = 0
    const client = fakeClient({
      decide: async () => {
        attempts += 1
        throw new AgentClientError('network_error')
      },
    })
    const conversation = bind(client, storage)
    await conversation.recoverAction()
    await conversation.decide(actionId, decision)
    const firstError = conversation.state.error
    conversation.dispose()

    const restored = bind(client, storage)
    await restored.recoverAction()
    await restored.decide(actionId, decision)
    assert.equal(attempts, 0, 'no POST is allowed without a durable lock, including after restoration')
    assert.equal(firstError.code, 'action_storage_unavailable')
    assert.equal(restored.state.error.code, 'action_storage_unavailable')
    assert.equal(firstError.status, 0)
    assert.ok(!firstError.message.includes(privateFailure))
    assert.equal('cause' in firstError, false)
    assert.equal(JSON.parse(storage.getItem(key)).action_locked, false)
  })

  test(`missing, unreadable or silently unwritable storage blocks ${decision} before its POST`, async () => {
    const variants = [
      ['missing storage', () => undefined],
      ['missing methods', () => ({})],
      ['missing writer', () => ({ getItem: () => null })],
      ['missing reader', () => ({ setItem() {} })],
      ['failed reader', () => ({ getItem() { throw new Error(privateFailure) }, setItem() {} })],
      ['failed writer', () => ({ getItem: () => null, setItem() { throw new Error(privateFailure) } })],
      ['silent writer', () => {
        const storage = memoryStorage()
        seedPending(storage)
        storage.setItem = () => {}
        return storage
      }],
    ]
    for (const [name, makeStorage] of variants) {
      let attempts = 0
      const client = fakeClient({
        chat: async () => chat({ ...action(), model_rounds: 0 }),
        decide: async () => {
          attempts += 1
          return action({ status: 'cancelled', pending_action: null, action_result: result('cancelled') })
        },
      })
      const conversation = bind(client, makeStorage())
      if (conversation.state.actionId) await conversation.recoverAction()
      else await conversation.send('Prepare action')
      await conversation.decide(actionId, decision)
      assert.equal(attempts, 0, name)
      assert.equal(conversation.state.error.code, 'action_storage_unavailable', name)
      assert.ok(!conversation.state.error.message.includes(privateFailure), name)
    }
  })

  test(`successful durable ${decision} locking precedes an uncertain POST and survives restoration`, async () => {
    const storage = memoryStorage()
    let attempts = 0
    let lockAtPost
    const client = fakeClient({
      chat: async () => chat({ ...action(), model_rounds: 0 }),
      decide: async () => {
        attempts += 1
        lockAtPost = JSON.parse(storage.getItem(key))
        throw new AgentClientError('network_error')
      },
    })
    const conversation = bind(client, storage)
    await conversation.send('Prepare action')
    await conversation.decide(actionId, decision)
    assert.equal(lockAtPost.action_id, actionId)
    assert.equal(lockAtPost.user_id, identity.user_id)
    assert.equal(lockAtPost.action_locked, true)
    conversation.dispose()

    const restored = bind(client, storage)
    await restored.recoverAction()
    await restored.decide(actionId, 'confirm')
    await restored.decide(actionId, 'cancel')
    assert.equal(attempts, 1)
    assert.equal(restored.state.recoveryOnly, true)
    assert.equal(restored.state.canDecide, false)
  })

  test(`storage failure after a saved ${decision} lock keeps all uncertain outcomes GET-only`, async () => {
    for (const status of ['network_error', 'executing', 'unknown']) {
      const storage = memoryStorage()
      let attempts = 0
      const client = fakeClient({
        chat: async () => chat({ ...action(), model_rounds: 0 }),
        decide: async () => {
          attempts += 1
          assert.equal(JSON.parse(storage.getItem(key)).action_locked, true)
          storage.setItem = () => { throw new Error(privateFailure) }
          if (status === 'network_error') throw new AgentClientError('network_error')
          return action({ status, pending_action: null, action_result: result(status) })
        },
      })
      const conversation = bind(client, storage)
      await conversation.send('Prepare action')
      await conversation.decide(actionId, decision)
      conversation.dispose()

      const restored = bind(client, storage)
      await restored.recoverAction()
      await restored.decide(actionId, 'confirm')
      await restored.decide(actionId, 'cancel')
      assert.equal(attempts, 1, status)
      assert.equal(JSON.parse(storage.getItem(key)).action_locked, true, status)
      assert.equal(restored.state.recoveryOnly, true, status)
      assert.equal(restored.state.canDecide, false, status)
    }
  })
}

test('a written lock that cannot be read back cannot authorize a decision POST', async () => {
  const storage = memoryStorage()
  const getItem = storage.getItem
  storage.getItem = cacheKey => {
    const raw = getItem(cacheKey)
    if (raw && JSON.parse(raw).action_locked) throw new Error(privateFailure)
    return raw
  }
  let attempts = 0
  const client = fakeClient({
    chat: async () => chat({ ...action(), model_rounds: 0 }),
    decide: async () => {
      attempts += 1
      return action({ status: 'cancelled', pending_action: null, action_result: result('cancelled') })
    },
  })
  const conversation = bind(client, storage)
  await conversation.send('Prepare action')
  await conversation.decide(actionId, 'confirm')
  assert.equal(attempts, 0)
  assert.equal(conversation.state.error.code, 'action_storage_unavailable')
  assert.equal(JSON.parse(storage.values.get(key)).action_locked, true)
})

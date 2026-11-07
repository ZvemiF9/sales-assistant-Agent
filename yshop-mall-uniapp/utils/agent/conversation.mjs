import {
  AgentClientError, isAgentId, isSafeProductId, publicAgentError,
  validateAgentAction, validateAgentChat, validateChatRequest, validateUserIdentity,
} from './client.mjs'

const CACHE_PREFIX = 'yshop-agent:conversation:v1:user:'
const MAX_MESSAGES = 12
const MAX_CACHE_BYTES = 65536
const CART_UNTRUSTED = /规则|政策|描述|指令|提示词|系统|如果|假如|若|不要|别|不加|取消|[?？]|吗|是否|能否|能不能|可以.*购物车|["'`“”‘’「」『』]/
const NUMERIC_LABEL_LETTERS = new Set(Array.from('零〇一二两三四五六七八九十百千万点负正eE到至件个台份只'))

function frozen(value) {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(frozen)
    Object.freeze(value)
  }
  return value
}

function bytes(value) {
  return new TextEncoder().encode(value).byteLength
}

function cachedText(value, role) {
  return typeof value === 'string' && Array.from(value).length <= (role === 'user' ? 1000 : 32768)
    && !/Bearer\s+\S+|(?:authorization|access_token|api_key)\s*[:=]/i.test(value)
}

function strippableLabel(value) {
  return Array.from(value).some(letter => /\p{L}/u.test(letter) && !NUMERIC_LABEL_LETTERS.has(letter))
}

function initialState() {
  return {
    identity: null, authStatus: 'unverified', sessionId: null, messages: [],
    candidates: [], candidateRevision: 0, sources: [], notice: null, status: 'idle',
    pendingAction: null, actionResult: null, actionId: null, actionVerified: false,
    recoveryOnly: false, canDecide: false, busy: false, phase: 'idle', error: null, streamStatus: null,
  }
}

export function createConversation({ client, storage, now = Date.now }) {
  const state = initialState()
  let ownerId = null
  let generation = 0
  let active = null
  let disposed = false
  const snapshots = new Map()
  const consumed = new Set()

  function storageCall(method, ...args) {
    try { return storage && typeof storage[method] === 'function' ? storage[method](...args) : null } catch { return null }
  }

  function cacheKey() {
    return ownerId === null ? null : `${CACHE_PREFIX}${ownerId}`
  }

  function dropCache() {
    const key = cacheKey()
    if (key) storageCall('removeItem', key)
  }

  function syncDecision() {
    state.canDecide = !disposed && !state.busy && state.authStatus === 'authenticated'
      && state.actionVerified && !state.recoveryOnly && state.pendingAction !== null
      && !consumed.has(state.actionId) && state.pendingAction.expires_at * 1000 > now()
  }

  function invalidate() {
    generation += 1
    if (active) active.abort()
    active = null
    state.busy = false
    state.phase = 'idle'
    state.streamStatus = null
    syncDecision()
  }

  function resetVisible() {
    const revision = state.candidateRevision + 1
    Object.assign(state, initialState(), { candidateRevision: revision })
    snapshots.clear()
    consumed.clear()
  }

  function cachePayload() {
    return {
      version: 1, user_id: ownerId, session_id: state.sessionId,
      messages: state.messages.map(({ role, content }) => ({ role, content })),
      action_id: state.actionId, action_locked: state.recoveryOnly || consumed.has(state.actionId),
    }
  }

  function save() {
    if (ownerId === null || disposed) return false
    state.messages = state.messages.slice(-MAX_MESSAGES)
    let serialized = JSON.stringify(cachePayload())
    while (bytes(serialized) > MAX_CACHE_BYTES && state.messages.length) {
      state.messages = state.messages.slice(1)
      serialized = JSON.stringify(cachePayload())
    }
    try {
      if (!storage || typeof storage.setItem !== 'function' || typeof storage.getItem !== 'function') return false
      const key = cacheKey()
      storage.setItem(key, serialized)
      // The uni-app synchronous writer can suppress errors, so verify the stored value.
      return storage.getItem(key) === serialized
    } catch {
      return false
    }
  }

  function restore() {
    const raw = storageCall('getItem', cacheKey())
    if (!raw) return
    try {
      if (typeof raw !== 'string' || bytes(raw) > MAX_CACHE_BYTES) throw new Error()
      const value = JSON.parse(raw)
      const allowed = ['version', 'user_id', 'session_id', 'messages', 'action_id', 'action_locked']
      if (!value || typeof value !== 'object' || Object.keys(value).length !== allowed.length || Object.keys(value).some(key => !allowed.includes(key))
        || value.version !== 1 || value.user_id !== ownerId
        || (value.session_id !== null && !isAgentId(value.session_id))
        || (value.action_id !== null && (!isAgentId(value.action_id) || value.session_id === null))
        || typeof value.action_locked !== 'boolean' || (value.action_id === null && value.action_locked)
        || !Array.isArray(value.messages) || value.messages.length > MAX_MESSAGES) throw new Error()
      const messages = value.messages.map(message => {
        if (!message || typeof message !== 'object' || Object.keys(message).length !== 2
          || !['user', 'assistant'].includes(message.role) || !cachedText(message.content, message.role)) throw new Error()
        return { role: message.role, content: message.content }
      })
      state.messages = messages
      state.sessionId = value.session_id
      state.actionId = value.action_id
      state.recoveryOnly = value.action_locked
      state.actionVerified = false
      if (state.actionId) state.status = 'recovery_required'
      if (value.action_locked) consumed.add(state.actionId)
    } catch {
      dropCache()
    }
  }

  function bindIdentity(identity) {
    if (disposed) return null
    const verified = validateUserIdentity(identity)
    invalidate()
    if (ownerId !== null && ownerId !== verified.user_id) dropCache()
    ownerId = verified.user_id
    resetVisible()
    state.identity = frozen(verified)
    state.authStatus = 'authenticated'
    restore()
    syncDecision()
    return state.identity
  }

  function clearIdentity() {
    invalidate()
    dropCache()
    ownerId = null
    resetVisible()
    state.authStatus = 'login_required'
    state.error = new AgentClientError('login_required', 401)
  }

  function setError(error) {
    const publicError = publicAgentError(error)
    if (publicError.code === 'login_required' || publicError.status === 401) clearIdentity()
    else state.error = publicError
    return null
  }

  function begin(phase) {
    generation += 1
    active = new AbortController()
    state.busy = true
    state.phase = phase
    state.error = null
    syncDecision()
    return { generation, controller: active }
  }

  function current(operation) {
    return !disposed && operation.generation === generation
  }

  function finish(operation) {
    if (!current(operation)) return
    active = null
    state.busy = false
    state.phase = 'idle'
    state.streamStatus = null
    syncDecision()
  }

  async function verifyIdentity() {
    if (disposed) return null
    invalidate()
    resetVisible()
    state.authStatus = 'checking'
    const operation = begin('identity')
    try {
      const identity = await client.me({ signal: operation.controller.signal })
      if (!current(operation)) return null
      return bindIdentity(identity)
    } catch (error) {
      if (!current(operation)) return null
      state.authStatus = 'unverified'
      return setError(error)
    } finally {
      finish(operation)
    }
  }

  function applyAction(response) {
    if (state.sessionId && response.session_id !== state.sessionId) throw new AgentClientError('session_conflict', 409)
    const pending = response.pending_action
    const outcome = response.action_result
    if (pending) {
      const previous = snapshots.get(pending.action_id)
      if (previous && JSON.stringify(previous) !== JSON.stringify(pending)) throw new AgentClientError('agent_invalid_response')
      const snapshot = previous || frozen(pending)
      snapshots.set(snapshot.action_id, snapshot)
      const changedId = state.actionId !== snapshot.action_id
      state.actionId = snapshot.action_id
      state.pendingAction = snapshot
      state.actionResult = null
      state.actionVerified = true
      state.recoveryOnly = changedId ? consumed.has(snapshot.action_id) : state.recoveryOnly || consumed.has(snapshot.action_id)
    } else if (outcome) {
      state.actionId = outcome.action_id
      state.actionResult = frozen(outcome)
      state.pendingAction = null
      state.actionVerified = true
      state.recoveryOnly = ['executing', 'unknown'].includes(outcome.status)
      consumed.add(outcome.action_id)
    } else {
      state.actionId = null
      state.pendingAction = null
      state.actionResult = null
      state.actionVerified = false
      state.recoveryOnly = false
    }
    state.sessionId = response.session_id
    state.status = response.status
    syncDecision()
  }

  function authenticated() {
    if (!disposed && state.authStatus === 'authenticated' && state.identity !== null) return true
    setError(new AgentClientError('login_required', 401))
    return false
  }

  async function send(message, options = {}) {
    if (disposed || state.busy || !authenticated()) return null
    if (state.actionId && (state.pendingAction || !state.actionVerified || state.recoveryOnly)) return setError(new AgentClientError('action_conflict', 409))
    let input
    try {
      input = validateChatRequest(state.sessionId ? { message, session_id: state.sessionId } : { message })
      if (!cachedText(input.message, 'user')) throw new AgentClientError('invalid_request')
    } catch (error) { return setError(error) }
    const operation = begin('chat')
    state.candidateRevision += 1
    state.candidates = []
    state.sources = []
    state.notice = null
    const userMessage = { role: 'user', content: input.message }
    state.messages = [...state.messages, userMessage].slice(-MAX_MESSAGES)
    try {
      const response = await client.chat(input, {
        ...options, signal: operation.controller.signal,
        onEvent(event) {
          if (!current(operation)) return
          if (event.event === 'status') state.streamStatus = event.data.status
        },
      })
      if (!current(operation)) return null
      const verified = validateAgentChat(response)
      applyAction(verified)
      state.candidates = frozen(verified.products)
      state.sources = frozen(verified.sources)
      state.notice = verified.notice
      state.messages = [...state.messages, { role: 'assistant', content: verified.answer }]
      save()
      return verified
    } catch (error) {
      if (!current(operation)) return null
      state.messages = state.messages.filter(item => item !== userMessage)
      return setError(error)
    } finally {
      finish(operation)
    }
  }

  async function prepare({ productId, uniqueId, quantity, revision } = {}, options = {}) {
    if (disposed || state.busy || !authenticated()) return null
    if (!isSafeProductId(productId) || typeof quantity !== 'number' || !Number.isSafeInteger(quantity) || quantity < 1 || quantity > 9999
      || typeof uniqueId !== 'string' || !uniqueId.trim() || uniqueId.length > 128) return setError(new AgentClientError('invalid_request'))
    const index = state.candidates.findIndex(item => item.product_id === productId)
    if (!Number.isSafeInteger(revision) || revision !== state.candidateRevision || index < 0) return setError(new AgentClientError('stale_candidate'))
    const candidate = state.candidates[index]
    const selected = candidate.skus.find(item => item.unique_id === uniqueId)
    if (!selected) return setError(new AgentClientError('sku_unavailable'))
    if (selected.stock < quantity) return setError(new AgentClientError('insufficient_stock'))
    const label = selected.name
    const message = `第${index + 1}个 ${label}规格，加入${quantity}件购物车`
    let quantityClause = message.replace(/第[0-9零〇一二两三四五六七八九十百千万]+[个件款台]?/g, ' ')
    for (const name of [candidate.name, label]) {
      if (name && strippableLabel(name)) quantityClause = quantityClause.split(name).join(' ')
    }
    // Match the server's conservative SKU matching and quantity-clause grammar.
    if (!strippableLabel(label) || quantityClause.trim() !== `规格，加入${quantity}件购物车`
      || CART_UNTRUSTED.test(message) || /第[0-9零〇一二两三四五六七八九十百千万]/.test(label)
      || candidate.skus.filter(item => item.name && message.includes(item.name)).length !== 1
      || label.length > 500) return setError(new AgentClientError('sku_selection'))
    return send(message, options)
  }

  async function decide(actionId, decision) {
    if (disposed || state.busy || !authenticated()) return null
    if (!isAgentId(actionId) || !['confirm', 'cancel'].includes(decision)) return setError(new AgentClientError('invalid_request'))
    if (state.pendingAction && state.pendingAction.expires_at * 1000 <= now()) {
      syncDecision()
      return setError(new AgentClientError('action_expired', 409))
    }
    if (actionId !== state.actionId || !state.canDecide || consumed.has(actionId)) return setError(new AgentClientError('action_conflict', 409))
    consumed.add(actionId)
    state.recoveryOnly = true
    const operation = begin('decision')
    try {
      if (!save()) return setError(new AgentClientError('action_storage_unavailable'))
      const response = await client.decide(actionId, decision, { signal: operation.controller.signal })
      if (!current(operation)) return null
      const verified = validateAgentAction(response)
      if (!verified.pending_action && !verified.action_result
        || verified.action_result && verified.action_result.action_id !== actionId) throw new AgentClientError('agent_invalid_response')
      applyAction(verified)
      state.messages = [...state.messages, { role: 'assistant', content: verified.answer }]
      save()
      return verified
    } catch (error) {
      if (!current(operation)) return null
      state.recoveryOnly = true
      state.actionVerified = false
      save()
      return setError(error)
    } finally {
      finish(operation)
    }
  }

  async function recoverAction(actionId = state.actionId) {
    if (disposed || state.busy || !authenticated()) return null
    if (!isAgentId(actionId) || actionId !== state.actionId) return setError(new AgentClientError('invalid_request'))
    const operation = begin('recovery')
    state.actionVerified = false
    try {
      const response = await client.getAction(actionId, { signal: operation.controller.signal })
      if (!current(operation)) return null
      const verified = validateAgentAction(response)
      if (!verified.pending_action && !verified.action_result
        || verified.pending_action && verified.pending_action.action_id !== actionId
        || verified.action_result && verified.action_result.action_id !== actionId) throw new AgentClientError('agent_invalid_response')
      applyAction(verified)
      state.messages = [...state.messages, { role: 'assistant', content: verified.answer }]
      save()
      return verified
    } catch (error) {
      if (!current(operation)) return null
      return setError(error)
    } finally {
      finish(operation)
    }
  }

  function newSession() {
    if (disposed) return
    const identity = state.identity
    const authStatus = state.authStatus
    invalidate()
    dropCache()
    resetVisible()
    state.identity = identity
    state.authStatus = authStatus
  }

  function dispose() {
    if (disposed) return
    save()
    disposed = true
    invalidate()
    resetVisible()
    ownerId = null
  }

  return Object.freeze({
    state, verifyIdentity, bindIdentity, clearIdentity, send, prepare, decide, recoverAction, newSession, dispose,
  })
}

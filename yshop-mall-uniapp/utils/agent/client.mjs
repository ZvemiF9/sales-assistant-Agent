const ID = /^[0-9a-f]{32}$/
const MAX_RESPONSE_BYTES = 1024 * 1024
const CHAT_STATUSES = ['answered', 'needs_input', 'no_evidence', 'limit_reached', 'awaiting_confirmation', 'executing', 'succeeded', 'failed', 'unknown', 'cancelled']
const RESULT_STATUSES = ['executing', 'succeeded', 'failed', 'unknown', 'cancelled']
const CART_PATH = '/pages/shop/ShoppingCart/index'

const PUBLIC_MESSAGES = Object.freeze({
  invalid_configuration: '导购服务地址配置无效。',
  invalid_request: '请检查输入，消息为1至1000字，数量为1至9999的整数。',
  network_error: '网络连接未完成，请检查连接后手动操作。',
  request_aborted: '本次请求已停止。',
  agent_unavailable: '导购服务暂时不可用，请稍后手动重试。',
  agent_invalid_response: '导购服务返回的数据无效，本次结果未被接受。',
  stale_candidate: '商品列表已更新，请重新选择当前商品。',
  login_required: '请先登录商城，或重新登录后继续。',
  action_not_found: '动作不存在或无法访问。',
  action_conflict: '动作状态已更新，请读取状态，勿重复提交。',
  action_storage_unavailable: '本次决定未提交。请恢复本地存储后刷新页面，再读取动作状态。',
  action_expired: '确认已过期，请重新查询并准备新的确认卡。',
  session_not_found: '会话不存在或无法访问，请新建会话。',
  session_busy: '此会话正在处理另一条消息，请稍后手动重试。',
  session_conflict: '会话状态已更新，请读取动作状态或新建会话。',
  session_unavailable: '会话保存暂时不可用，请稍后手动重试。',
  model_not_configured: '导购模型暂未就绪，请稍后继续。',
  model_timeout: '导购模型响应超时，请稍后手动重试。',
  model_unavailable: '导购模型暂时无法访问，请稍后手动重试。',
  model_rejected: '导购模型未能完成本次请求。',
  model_invalid_response: '导购模型返回的数据无效，请稍后手动重试。',
  rag_not_built: '演示规则暂未就绪。',
  rag_index_incompatible: '演示规则暂不可用。',
  rag_index_corrupt: '演示规则暂不可用。',
  rag_model_unavailable: '演示规则检索暂不可用。',
  rag_unavailable: '演示规则检索暂不可用。',
  shop_unavailable: '商城暂时无法访问，请稍后手动重试。',
  shop_invalid_response: '商城返回的数据无效。',
  shop_rejected: '商城未能完成本次请求。',
  unsupported_product: '当前仅支持普通商品。',
  cart_result_unknown: '加购结果暂时无法确认，请查看购物车，勿重复提交。',
  sku_unavailable: '该规格不属于此商品或已失效。',
  sku_selection: '该规格名称无法明确识别，请在商品详情中选择规格。',
  insufficient_stock: '当前规格库存不足，已有购物车数量也计入校验。',
  cart_state_ambiguous: '购物车存在多条匹配记录，请检查商城购物车。',
  cart_item_invalid: '该规格已有失效购物车记录，请检查商城购物车。',
  cart_verification_failed: '写后购物车结果未能确认，请查看购物车，勿重复提交。',
})

export class AgentClientError extends Error {
  constructor(code, status = 0, requestId) {
    const publicCode = Object.prototype.hasOwnProperty.call(PUBLIC_MESSAGES, code) ? code : 'agent_unavailable'
    super(PUBLIC_MESSAGES[publicCode])
    this.name = 'AgentClientError'
    this.code = publicCode
    this.status = Number.isInteger(status) && status >= 0 && status <= 599 ? status : 0
    if (typeof requestId === 'string' && ID.test(requestId)) this.requestId = requestId
  }
}

export function publicAgentError(error) {
  return error instanceof AgentClientError
    ? new AgentClientError(error.code, error.status, error.requestId)
    : new AgentClientError('agent_unavailable')
}

function check(condition) {
  if (!condition) throw new AgentClientError('agent_invalid_response')
}

function object(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function text(value, max = 32768, nonblank = false) {
  check(typeof value === 'string' && Array.from(value).length <= max && (!nonblank || value.trim().length > 0))
  return value
}

function integer(value, min = 0, max = Number.MAX_SAFE_INTEGER) {
  check(typeof value === 'number' && Number.isSafeInteger(value) && value >= min && value <= max)
  return value
}

export function isSafeProductId(value) {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0
}

export function isAgentId(value) {
  return typeof value === 'string' && ID.test(value)
}

function identifier(value) {
  check(isAgentId(value))
  return value
}

function list(value, max, mapper) {
  check(Array.isArray(value) && value.length <= max)
  return value.map(mapper)
}

function optionalText(value, max = 32768) {
  return value === null ? null : text(value, max)
}

function money(value, places) {
  check(typeof value === 'string' || typeof value === 'number')
  check(typeof value !== 'number' || Number.isFinite(value))
  const raw = String(value)
  check(/^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(raw) && raw.length <= 128)
  check(Number.isFinite(Number(raw)) && Number(raw) >= 0)
  if (places === undefined) return raw
  check(/^\d+(?:\.\d{1,2})?$/.test(raw))
  const [whole, fraction = ''] = raw.split('.')
  const normalizedWhole = whole.replace(/^0+(?=\d)/, '')
  check(normalizedWhole.length <= places)
  return `${normalizedWhole}.${fraction.padEnd(2, '0')}`
}

function cents(value) {
  return BigInt(value.replace('.', ''))
}

function image(value) {
  const raw = text(value, 2048)
  if (!raw) return ''
  if (raw.startsWith('/') && !raw.startsWith('//') && !/[?#\\]/.test(raw)) return raw
  try {
    const url = new URL(raw)
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return ''
    for (const key of url.searchParams.keys()) {
      if (/token|authorization|api.?key|password|secret/i.test(key)) return ''
    }
    return raw
  } catch {
    return ''
  }
}

function sku(value) {
  check(object(value))
  return {
    name: text(value.name, 16384),
    unique_id: text(value.unique_id, 128, true),
    price: money(value.price),
    stock: integer(value.stock),
    image: image(value.image),
  }
}

function product(value) {
  check(object(value))
  const skus = list(value.skus, 64, sku)
  check(new Set(skus.map(item => item.unique_id)).size === skus.length)
  return {
    product_id: integer(value.product_id, 1),
    name: text(value.name, 16384, true),
    description: text(value.description, 16384),
    image: image(value.image),
    skus,
  }
}

function source(value) {
  check(object(value) && value.kind === 'demo' && typeof value.score === 'number' && Number.isFinite(value.score) && value.score >= -1 && value.score <= 1)
  return {
    document_id: text(value.document_id, 1024),
    title: text(value.title, 16384),
    section: text(value.section, 16384),
    source: text(value.source, 2048),
    version: text(value.version, 1024),
    kind: 'demo',
    chunk_id: text(value.chunk_id, 1024),
    text: text(value.text, 32768),
    score: value.score,
  }
}

export function validateUserIdentity(value) {
  check(object(value))
  return { user_id: integer(value.user_id, 1), nickname: text(value.nickname, 16384) }
}

export function validateActionCard(value) {
  check(object(value) && value.kind === 'add_to_cart')
  const quantity = integer(value.quantity, 1, 9999)
  const unitPrice = money(value.unit_price, 14)
  const totalPrice = money(value.total_price, 18)
  check(cents(unitPrice) * BigInt(quantity) === cents(totalPrice))
  check(typeof value.expires_at === 'number' && Number.isFinite(value.expires_at) && value.expires_at > 0 && value.expires_at <= 253402300799)
  return {
    kind: 'add_to_cart',
    action_id: identifier(value.action_id),
    session_id: identifier(value.session_id),
    product_id: integer(value.product_id, 1),
    product_name: text(value.product_name, 500, true),
    unique_id: text(value.unique_id, 128, true),
    sku_name: text(value.sku_name, 500, true),
    quantity,
    unit_price: unitPrice,
    total_price: totalPrice,
    image: image(value.image),
    expires_at: value.expires_at,
  }
}

function knownError(value) {
  if (value === null) return null
  check(typeof value === 'string' && Object.prototype.hasOwnProperty.call(PUBLIC_MESSAGES, value))
  return value
}

export function validateActionResult(value) {
  check(object(value) && RESULT_STATUSES.includes(value.status) && value.cart_path === CART_PATH)
  const cartId = value.cart_id === null ? null : integer(value.cart_id, 1)
  const errorCode = knownError(value.error_code)
  check((value.status === 'succeeded') === (cartId !== null))
  check(!['succeeded', 'executing'].includes(value.status) || errorCode === null)
  return {
    action_id: identifier(value.action_id),
    session_id: identifier(value.session_id),
    status: value.status,
    cart_id: cartId,
    error_code: errorCode,
    cart_path: CART_PATH,
  }
}

function actionFields(value) {
  const sessionId = identifier(value.session_id)
  const pending = value.pending_action === null ? null : validateActionCard(value.pending_action)
  const outcome = value.action_result === null ? null : validateActionResult(value.action_result)
  check(pending === null || pending.session_id === sessionId)
  check(outcome === null || outcome.session_id === sessionId)
  if (value.status === 'awaiting_confirmation') check(pending !== null && outcome === null)
  else if (RESULT_STATUSES.includes(value.status)) check(pending === null && outcome !== null && outcome.status === value.status)
  else check(pending === null && outcome === null)
  return { session_id: sessionId, pending_action: pending, action_result: outcome }
}

export function validateAgentAction(value) {
  check(object(value) && ['awaiting_confirmation', 'needs_input', ...RESULT_STATUSES].includes(value.status))
  return { ...actionFields(value), status: value.status, answer: text(value.answer, 4000, true) }
}

function ownedAction(value, actionId, allowReplacement = false) {
  const response = validateAgentAction(value)
  check(response.pending_action !== null || response.action_result !== null)
  if (response.pending_action) check(allowReplacement || response.pending_action.action_id === actionId)
  if (response.action_result) check(response.action_result.action_id === actionId)
  return response
}

export function validateAgentChat(value) {
  check(object(value) && CHAT_STATUSES.includes(value.status) && object(value.usage))
  const products = list(value.products, 3, product)
  check(new Set(products.map(item => item.product_id)).size === products.length)
  return {
    ...actionFields(value),
    answer: text(value.answer),
    status: value.status,
    products,
    sources: list(value.sources, 64, source),
    notice: optionalText(value.notice),
    tool_calls: list(value.tool_calls, 64, item => {
      check(object(item) && /^[A-Za-z0-9_-]{1,64}$/.test(item.name) && ['succeeded', 'failed'].includes(item.status))
      return { name: item.name, status: item.status, elapsed_ms: integer(item.elapsed_ms) }
    }),
    model_rounds: integer(value.model_rounds, 0, 5),
    usage: {
      input_tokens: value.usage.input_tokens === null ? null : integer(value.usage.input_tokens),
      output_tokens: value.usage.output_tokens === null ? null : integer(value.usage.output_tokens),
    },
  }
}

function validateStreamEvent({ event, data }) {
  check(object(data))
  switch (event) {
    case 'status':
      check(data.status === 'processing')
      return { event, data: { status: 'processing' } }
    case 'text': return { event, data: { text: text(data.text) } }
    case 'products': {
      const products = list(data.products, 3, product)
      check(new Set(products.map(item => item.product_id)).size === products.length)
      return { event, data: { products } }
    }
    case 'sources': return { event, data: { sources: list(data.sources, 64, source), notice: optionalText(data.notice) } }
    case 'confirmation': return { event, data: { pending_action: validateActionCard(data.pending_action) } }
    case 'done': return { event, data: validateAgentChat(data) }
    case 'error':
      check(object(data.error))
      throw new AgentClientError(data.error.code, data.error.code === 'login_required' ? 401 : 0)
    default: throw new AgentClientError('agent_invalid_response')
  }
}

function validateHistory(value) {
  check(object(value))
  const actionId = identifier(value.action_id)
  const events = list(value.events, 3, event => {
    check(object(event) && ['legacy', 'prepared', 'executing', 'finished', 'replaced'].includes(event.kind))
    check(['pending', ...RESULT_STATUSES].includes(event.status))
    check(typeof event.occurred_at === 'number' && Number.isFinite(event.occurred_at) && event.occurred_at > 0 && event.occurred_at <= 253402300799)
    const related = event.related_action_id === null ? null : identifier(event.related_action_id)
    const requestId = event.request_id === null ? null : identifier(event.request_id)
    const errorCode = knownError(event.error_code)
    check(event.kind !== 'prepared' || event.status === 'pending')
    check(event.kind !== 'executing' || event.status === 'executing')
    check(event.kind !== 'replaced' || (event.status === 'cancelled' && related !== null))
    check(event.kind !== 'finished' || ['succeeded', 'failed', 'unknown', 'cancelled'].includes(event.status))
    check(['prepared', 'replaced'].includes(event.kind) || related === null)
    check(event.kind === 'finished' || errorCode === null)
    check(event.status !== 'succeeded' || errorCode === null)
    check(event.kind !== 'legacy' || requestId === null)
    return {
      sequence: integer(event.sequence, 1, 3), kind: event.kind, status: event.status,
      occurred_at: event.occurred_at, request_id: requestId, related_action_id: related, error_code: errorCode,
    }
  })
  check(events.length >= 1)
  events.forEach((event, index) => {
    check(event.sequence === index + 1 && event.related_action_id !== actionId)
    if (index === 0) check(['prepared', 'legacy'].includes(event.kind))
    else {
      const previous = events[index - 1].status
      const allowed = previous === 'pending' ? ['executing', 'finished', 'replaced'] : previous === 'executing' ? ['finished'] : []
      check(allowed.includes(event.kind))
      check(previous !== 'pending' || !['succeeded', 'unknown'].includes(event.status))
      check(previous !== 'executing' || event.status !== 'cancelled')
    }
  })
  return { action_id: actionId, session_id: identifier(value.session_id), events }
}

export function validateChatRequest(value) {
  if (!object(value) || Object.keys(value).some(key => !['message', 'session_id'].includes(key)) || typeof value.message !== 'string') {
    throw new AgentClientError('invalid_request')
  }
  const message = value.message.trim()
  const length = Array.from(message).length
  if (length < 1 || length > 1000) throw new AgentClientError('invalid_request')
  if (value.session_id !== undefined && value.session_id !== null && !isAgentId(value.session_id)) throw new AgentClientError('invalid_request')
  return value.session_id == null ? { message } : { message, session_id: value.session_id }
}

function rejectCredential(value, token) {
  if (typeof value === 'string' && value.includes(token)) throw new AgentClientError('agent_invalid_response')
  if (Array.isArray(value)) value.forEach(item => rejectCredential(item, token))
  else if (object(value)) Object.values(value).forEach(item => rejectCredential(item, token))
}

async function boundedResponseText(response) {
  const length = response.headers.get('Content-Length')
  check(length === null || (/^\d+$/.test(length) && Number(length) <= MAX_RESPONSE_BYTES))
  if (!response.body || typeof response.body.getReader !== 'function') {
    const value = await response.text()
    check(new TextEncoder().encode(value).byteLength <= MAX_RESPONSE_BYTES)
    return value
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8', { fatal: true })
  let bytes = 0
  let value = ''
  try {
    while (true) {
      const chunk = await reader.read()
      if (chunk.done) break
      bytes += chunk.value.byteLength
      if (bytes > MAX_RESPONSE_BYTES) {
        await reader.cancel()
        throw new AgentClientError('agent_invalid_response')
      }
      value += decoder.decode(chunk.value, { stream: true })
    }
    return value + decoder.decode()
  } finally {
    reader.releaseLock()
  }
}

export function createAgentClient({ baseUrl, getToken, fetchImpl = typeof fetch === 'function' ? fetch.bind(globalThis) : null }) {
  let base
  try {
    const url = new URL(baseUrl)
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error()
    base = url.href.replace(/\/+$/, '')
  } catch {
    throw new AgentClientError('invalid_configuration')
  }
  if (typeof getToken !== 'function' || typeof fetchImpl !== 'function') throw new AgentClientError('invalid_configuration')

  async function request(path, { method = 'GET', body, signal, validate, stream = false, onEvent }) {
    let token
    try { token = getToken() } catch { throw new AgentClientError('login_required', 401) }
    if (typeof token !== 'string' || token.length > 8192 || !/^[A-Za-z0-9._~+/-]+=*$/.test(token)) throw new AgentClientError('login_required', 401)
    if (body && body.message && body.message.includes(token)) throw new AgentClientError('invalid_request')
    const headers = { Authorization: `Bearer ${token}`, Accept: stream ? 'text/event-stream' : 'application/json' }
    if (body !== undefined) headers['Content-Type'] = 'application/json'
    let response
    try {
      response = await fetchImpl(`${base}${path}`, {
        method, headers, credentials: 'omit', cache: 'no-store', redirect: 'error',
        ...(body === undefined ? {} : { body: JSON.stringify(body) }), ...(signal ? { signal } : {}),
      })
    } catch {
      throw new AgentClientError(signal && signal.aborted ? 'request_aborted' : 'network_error')
    }
    const status = response.status
    const rawId = response.headers.get('X-Request-ID')
    const requestId = rawId !== token && isAgentId(rawId) ? rawId : undefined
    function sameIdentity() {
      let currentToken
      try { currentToken = getToken() } catch { throw new AgentClientError('login_required', 401) }
      if (currentToken !== token) throw new AgentClientError('login_required', 401)
    }
    if (stream && status >= 200 && status < 300) {
      let validationError
      try {
        check(/^text\/event-stream(?:;|$)/i.test(response.headers.get('Content-Type') || ''))
        const final = await readSSE(response.body, {
          signal, onEvent,
          validateEvent(value) {
            try {
              sameIdentity()
              rejectCredential(value.data, token)
              return validateStreamEvent(value)
            } catch (error) {
              validationError = publicAgentError(error)
              throw new SSEError()
            }
          },
        })
        sameIdentity()
        return final.data
      } catch (error) {
        if (validationError) throw new AgentClientError(validationError.code, validationError.status || status, requestId)
        if (error instanceof AgentClientError) throw new AgentClientError(error.code, error.status || status, requestId)
        throw new AgentClientError(error instanceof SSEError ? error.code : 'agent_invalid_response', status, requestId)
      }
    }
    let payload
    try {
      check(/^application\/(?:[a-z0-9.+-]+\+)?json(?:;|$)/i.test(response.headers.get('Content-Type') || ''))
      payload = JSON.parse(await boundedResponseText(response))
    } catch {
      if (status === 401) throw new AgentClientError('login_required', 401, requestId)
      if (status < 200 || status >= 300) throw new AgentClientError('agent_unavailable', status, requestId)
      throw new AgentClientError(signal && signal.aborted ? 'request_aborted' : 'agent_invalid_response', status, requestId)
    }
    sameIdentity()
    if (status < 200 || status >= 300) {
      const code = status === 401 ? 'login_required' : object(payload) && object(payload.error) ? payload.error.code : 'agent_unavailable'
      throw new AgentClientError(code, status, requestId)
    }
    try {
      rejectCredential(payload, token)
      return validate(payload)
    } catch {
      throw new AgentClientError('agent_invalid_response', status, requestId)
    }
  }

  function actionPath(actionId) {
    if (!isAgentId(actionId)) throw new AgentClientError('invalid_request')
    return `/agent/actions/${actionId}`
  }

  return Object.freeze({
    me: (options = {}) => request('/agent/me', { signal: options.signal, validate: validateUserIdentity }),
    chat: async (input, options = {}) => request(options.stream ? '/agent/chat/stream' : '/agent/chat', {
      method: 'POST', body: validateChatRequest(input), signal: options.signal, validate: validateAgentChat,
      stream: options.stream === true, onEvent: options.onEvent,
    }),
    getAction: async (actionId, options = {}) => request(actionPath(actionId), { signal: options.signal, validate: value => ownedAction(value, actionId) }),
    getHistory: async (actionId, options = {}) => request(`${actionPath(actionId)}/history`, {
      signal: options.signal,
      validate: value => {
        const history = validateHistory(value)
        check(history.action_id === actionId)
        return history
      },
    }),
    decide: async (actionId, decision, options = {}) => {
      const path = actionPath(actionId)
      if (!['confirm', 'cancel'].includes(decision)) throw new AgentClientError('invalid_request')
      return request(`${path}/decision`, { method: 'POST', body: { decision }, signal: options.signal, validate: value => ownedAction(value, actionId, true) })
    },
  })
}
import { readSSE, SSEError } from './sse.mjs'


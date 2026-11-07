export const sessionId = 'a'.repeat(32)
export const actionId = 'b'.repeat(32)
export const replacementId = 'c'.repeat(32)
export const requestId = 'd'.repeat(32)

export function product(overrides = {}) {
  return {
    product_id: 101,
    name: '轻便保温杯',
    description: '日常使用的保温杯。',
    image: 'https://example.com/cup.png',
    skus: [
      { name: '红色', unique_id: 'cup-red', price: '69.90', stock: 20, image: 'https://example.com/red.png' },
      { name: '绿色', unique_id: 'cup-green', price: '79.90', stock: 12, image: 'https://example.com/green.png' },
    ],
    ...overrides,
  }
}

export function card(overrides = {}) {
  return {
    kind: 'add_to_cart',
    action_id: actionId,
    session_id: sessionId,
    product_id: 101,
    product_name: '轻便保温杯',
    unique_id: 'cup-red',
    sku_name: '红色',
    quantity: 2,
    unit_price: '69.90',
    total_price: '139.80',
    image: 'https://example.com/red.png',
    expires_at: 1893456900,
    ...overrides,
  }
}

export function result(status = 'succeeded', overrides = {}) {
  return {
    action_id: actionId,
    session_id: sessionId,
    status,
    cart_id: status === 'succeeded' ? 501 : null,
    error_code: status === 'unknown' ? 'cart_result_unknown' : null,
    cart_path: '/pages/shop/ShoppingCart/index',
    ...overrides,
  }
}

export function chat(overrides = {}) {
  return {
    session_id: sessionId,
    answer: '请选择商品。',
    status: 'needs_input',
    products: [],
    sources: [],
    notice: null,
    tool_calls: [],
    model_rounds: 1,
    usage: { input_tokens: null, output_tokens: null },
    pending_action: null,
    action_result: null,
    ...overrides,
  }
}

export function action(overrides = {}) {
  return {
    session_id: sessionId,
    status: 'awaiting_confirmation',
    answer: '请确认是否加入购物车。',
    pending_action: card(),
    action_result: null,
    ...overrides,
  }
}

export function source(overrides = {}) {
  return {
    document_id: 'returns-demo',
    title: '售后演示规则',
    section: '退货条件',
    source: 'demo-returns.md',
    version: '2026-10-29',
    kind: 'demo',
    chunk_id: 'returns-1',
    text: '未使用且包装完好的商品，可按演示规则申请退货。',
    score: 0.8,
    ...overrides,
  }
}

export function history(overrides = {}) {
  return {
    action_id: actionId,
    session_id: sessionId,
    events: [{
      sequence: 1,
      kind: 'prepared',
      status: 'pending',
      occurred_at: 1893456000,
      request_id: requestId,
      related_action_id: null,
      error_code: null,
    }],
    ...overrides,
  }
}

export function jsonResponse(value, status = 200, headers = {}) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  })
}

export function deferred() {
  let resolve
  let reject
  const promise = new Promise((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

export function memoryStorage() {
  const values = new Map()
  return {
    values,
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: key => values.delete(key),
  }
}

export function fakeClient(overrides = {}) {
  const observed = []
  return {
    observed,
    me: async () => ({ user_id: 11, nickname: '测试用户' }),
    chat: async (input, options) => {
      observed.push({ method: 'chat', input, options })
      return chat({ products: [product()] })
    },
    getAction: async id => {
      observed.push({ method: 'getAction', id })
      return action()
    },
    getHistory: async id => {
      observed.push({ method: 'getHistory', id })
      return history()
    },
    decide: async (id, decision) => {
      observed.push({ method: 'decide', id, decision })
      return action({ status: 'cancelled', pending_action: null, action_result: result('cancelled') })
    },
    ...overrides,
  }
}

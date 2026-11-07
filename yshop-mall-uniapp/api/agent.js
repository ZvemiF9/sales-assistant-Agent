import { VUE_APP_AGENT_API_URL } from '@/config'
import cookie from '@/utils/store/cookie'
import { createAgentClient } from '@/utils/agent/client.mjs'

export function getAgentToken() {
  return cookie.get('login_status')
}

export function createShopAgentClient() {
  return createAgentClient({ baseUrl: VUE_APP_AGENT_API_URL, getToken: getAgentToken })
}

export const agentStorage = {
  getItem: key => {
    const value = cookie.get(key)
    return typeof value === 'string' ? value : null
  },
  setItem: (key, value) => cookie.set(key, value),
  removeItem: key => cookie.remove(key),
}

export function goToAgentLogin() {
  cookie.set('login_back_url', '/pages/agent/Chat/index')
  uni.navigateTo({ url: '/pages/user/Login/index' })
}

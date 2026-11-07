<template>
  <view v-if="conversationState" class="agent-page" :style="viewportStyle">
    <view class="agent-header">
      <button class="icon-button home-button" role="button" aria-label="返回商城" title="返回商城" @click="goHome">
        <uni-icons type="home" :size="23" color="#444444" />
      </button>
      <view class="header-copy">
        <text class="page-title">商城导购</text>
        <text class="session-label">{{ conversationState.sessionId ? '会话 ' + conversationState.sessionId.slice(0, 8) : '新会话' }}</text>
      </view>
      <view class="stream-control">
        <text>流式</text>
        <switch aria-label="流式响应" :checked="streamEnabled" :disabled="!streamSupported || conversationState.busy" color="#d62f26" @change="streamEnabled = $event.detail.value" />
      </view>
      <button class="icon-button" role="button" aria-label="新建会话" title="新建会话" :disabled="conversationState.authStatus !== 'authenticated'" @click="newSession">
        <uni-icons type="compose" :size="23" color="#d62f26" />
      </button>
    </view>

    <view v-if="conversationState.authStatus === 'checking'" class="state-bar" role="status" aria-live="polite">正在校验商城身份...</view>
    <view v-else-if="conversationState.authStatus !== 'authenticated'" class="identity-bar">
      <text>{{ conversationState.error ? conversationState.error.message : '请登录商城后继续。' }}</text>
      <button v-if="conversationState.authStatus === 'login_required' || !hasShopToken" class="login-button" role="button" aria-label="登录商城" @click="login">登录商城</button>
      <button v-else class="identity-button" role="button" aria-label="重新校验商城身份" :disabled="conversationState.busy" @click="checkIdentity">重新校验身份</button>
    </view>
    <view v-else-if="conversationState.error" class="error-bar" role="alert">
      <text>{{ conversationState.error.message }}</text>
      <text v-if="conversationState.error.requestId" class="request-id">请求编号 {{ conversationState.error.requestId }}</text>
    </view>
    <view v-else-if="conversationState.busy" class="state-bar" role="status" aria-live="polite">{{ processingText }}</view>

    <scroll-view class="chat-scroll" scroll-y :scroll-into-view="scrollAnchor" :scroll-with-animation="false" role="log" aria-live="polite" aria-label="导购对话">
      <view class="chat-content">
        <view v-if="!conversationState.messages.length && conversationState.authStatus === 'authenticated'" class="empty-chat">
          <uni-icons type="chat" :size="30" color="#d62f26" />
          <text>您好，今天想找什么商品？</text>
        </view>
        <view v-for="(message, index) in conversationState.messages" :key="index" class="message-row" :class="message.role">
          <view class="message-text">
            <text class="message-role">{{ message.role === 'user' ? '您' : '导购' }}</text>
            <text class="message-content">{{ message.content }}</text>
          </view>
        </view>
        <view v-if="conversationState.candidates.length" class="product-grid">
          <agent-product-card v-for="product in conversationState.candidates" :key="product.product_id + ':' + conversationState.candidateRevision" :product="product" :revision="conversationState.candidateRevision" :disabled="conversationState.busy || !!conversationState.actionId && !conversationState.actionResult" @prepare="prepare" @open="openProduct" />
        </view>
        <agent-action-card v-if="conversationState.actionId" :card="conversationState.pendingAction" :result="conversationState.actionResult" :action-id="conversationState.actionId" :verified="conversationState.actionVerified" :can-decide="conversationState.canDecide" :recovery-only="conversationState.recoveryOnly" :busy="conversationState.busy" :now-value="nowValue" @decision="decide" @recover="recover" @cart="goCart" />
        <agent-sources :sources="conversationState.sources" :notice="conversationState.notice" />
        <view id="agent-chat-end" class="chat-end" />
      </view>
    </scroll-view>

    <view class="agent-composer">
      <view class="composer-row">
        <textarea v-model="draft" class="message-input" aria-label="消息" placeholder="输入消息" :maxlength="-1" :auto-height="false" :adjust-position="false" :disabled="inputDisabled" />
        <button class="send-button" role="button" aria-label="发送消息" title="发送消息" :disabled="!canSend" @click="send">
          <uni-icons type="paperplane-filled" :size="23" color="#ffffff" />
        </button>
      </view>
      <text class="message-count" :class="{ 'invalid-count': messageLength > 1000 }">{{ messageLength }}/1000</text>
    </view>
  </view>
  <view v-else class="startup-error" role="alert">{{ startupError }}</view>
</template>

<script>
import UniIcons from '@/components/uni-icons/uni-icons.vue'
import AgentProductCard from '@/components/agent/AgentProductCard.vue'
import AgentActionCard from '@/components/agent/AgentActionCard.vue'
import AgentSources from '@/components/agent/AgentSources.vue'
import { agentStorage, createShopAgentClient, getAgentToken, goToAgentLogin } from '@/api/agent'
import { createConversation } from '@/utils/agent/conversation.mjs'
import { isSafeProductId, publicAgentError } from '@/utils/agent/client.mjs'

export default {
  name: 'AgentChat',
  components: { UniIcons, AgentProductCard, AgentActionCard, AgentSources },
  data() {
    const supported = typeof ReadableStream === 'function' && typeof TextDecoder === 'function' && typeof AbortController === 'function'
    return { conversationState: null, draft: '', scrollAnchor: '', viewportHeight: 0, nowValue: Date.now(), hasShopToken: false, startupError: '', streamSupported: supported, streamEnabled: supported }
  },
  computed: {
    messageLength() { return Array.from(this.draft.trim()).length },
    inputDisabled() {
      const state = this.conversationState
      return !state || state.authStatus !== 'authenticated' || state.busy
        || !!state.actionId && (!!state.pendingAction || !state.actionVerified || state.recoveryOnly)
    },
    canSend() { return !this.inputDisabled && this.messageLength >= 1 && this.messageLength <= 1000 },
    processingText() {
      const phase = this.conversationState.phase
      return phase === 'decision' ? '正在提交本次决定...' : phase === 'recovery' ? '正在读取动作状态...' : '正在处理消息...'
    },
    viewportStyle() { return this.viewportHeight ? { height: this.viewportHeight + 'px' } : {} },
  },
  watch: {
    'conversationState.messages'() { this.scrollToEnd() },
    'conversationState.candidates'() { this.scrollToEnd() },
    'conversationState.pendingAction'() { this.scrollToEnd() },
    'conversationState.authStatus'(value) { if (value !== 'authenticated') this.draft = '' },
  },
  onLoad() {
    this.createController()
    this._shown = false
    this._lastToken = getAgentToken()
    this._unsubscribeAgent = this.$store.subscribe(() => this.identityChanged())
    this._storageAgent = () => this.identityChanged()
    this._resizeAgent = () => {
      this.viewportHeight = window.visualViewport ? window.visualViewport.height : window.innerHeight
    }
    window.addEventListener('storage', this._storageAgent)
    window.addEventListener('resize', this._resizeAgent)
    if (window.visualViewport) window.visualViewport.addEventListener('resize', this._resizeAgent)
    this._resizeAgent()
    this._clockAgent = setInterval(() => { this.nowValue = Date.now() }, 1000)
  },
  onShow() {
    this._shown = true
    if (!this._conversation) this.createController()
    this.checkIdentity()
  },
  onHide() {
    this._shown = false
    this.draft = ''
    if (this._conversation) this._conversation.dispose()
    this._conversation = null
  },
  onUnload() { this.cleanup() },
  beforeDestroy() { this.cleanup() },
  methods: {
    createController() {
      try {
        this._conversation = createConversation({ client: createShopAgentClient(), storage: agentStorage })
        this.conversationState = this._conversation.state
      } catch (error) { this.startupError = publicAgentError(error).message }
    },
    identityChanged() {
      const token = getAgentToken()
      if (token === this._lastToken) return false
      this._lastToken = token
      this.hasShopToken = typeof token === 'string' && token.length > 0
      this.draft = ''
      if (this._conversation) this._conversation.clearIdentity()
      if (this._shown && this.hasShopToken) this.checkIdentity()
      return true
    },
    checkIdentity() {
      if (!this._conversation) return
      this._lastToken = getAgentToken()
      this.hasShopToken = typeof this._lastToken === 'string' && this._lastToken.length > 0
      if (!this.hasShopToken) this._conversation.clearIdentity()
      else this._conversation.verifyIdentity()
    },
    async send() {
      if (this.identityChanged()) return
      if (!this.canSend || !this._conversation) return
      const message = this.draft
      this.draft = ''
      await this._conversation.send(message, { stream: this.streamEnabled })
      this.scrollToEnd()
    },
    async prepare(selection) {
      if (this.identityChanged()) return
      if (!this._conversation) return
      await this._conversation.prepare(selection, { stream: this.streamEnabled })
      this.scrollToEnd()
    },
    async decide(decision) {
      if (this.identityChanged()) return
      if (!this._conversation || !this.conversationState.actionId) return
      await this._conversation.decide(this.conversationState.actionId, decision)
      this.scrollToEnd()
    },
    async recover() {
      if (this.identityChanged()) return
      if (!this._conversation) return
      await this._conversation.recoverAction()
      this.scrollToEnd()
    },
    newSession() {
      this.draft = ''
      if (this._conversation) this._conversation.newSession()
    },
    scrollToEnd() {
      this.scrollAnchor = ''
      this.$nextTick(() => { this.scrollAnchor = 'agent-chat-end' })
    },
    openProduct(productId) {
      if (!isSafeProductId(productId)) return
      uni.navigateTo({ url: '/pages/shop/GoodsCon/index?id=' + productId })
    },
    goHome() { uni.switchTab({ url: '/pages/home/index' }) },
    goCart() { uni.switchTab({ url: '/pages/shop/ShoppingCart/index' }) },
    login() { goToAgentLogin() },
    cleanup() {
      if (this._conversation) this._conversation.dispose()
      this._conversation = null
      if (this._unsubscribeAgent) this._unsubscribeAgent()
      this._unsubscribeAgent = null
      clearInterval(this._clockAgent)
      if (this._storageAgent) window.removeEventListener('storage', this._storageAgent)
      if (this._resizeAgent) {
        window.removeEventListener('resize', this._resizeAgent)
        if (window.visualViewport) window.visualViewport.removeEventListener('resize', this._resizeAgent)
      }
    },
  },
}
</script>

<style scoped>
.agent-page { width: 100%; height: 100vh; height: 100dvh; box-sizing: border-box; background: #ffffff; color: #333333; display: flex; flex-direction: column; font-size: 14px; letter-spacing: 0; overflow: hidden; }
.agent-header { flex: 0 0 auto; min-height: 68px; box-sizing: border-box; display: flex; align-items: center; gap: 12px; padding: 10px max(16px, env(safe-area-inset-right)) 10px max(16px, env(safe-area-inset-left)); border-bottom: 1px solid #e5e5e5; }
.icon-button { flex: 0 0 40px; width: 40px; height: 40px; min-height: 0; margin: 0; padding: 0; display: flex; align-items: center; justify-content: center; background: #f6f6f6; border-radius: 4px; line-height: 1; }
.header-copy { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.page-title { font-size: 20px; line-height: 1.4; font-weight: 600; overflow-wrap: anywhere; }
.session-label { font-size: 11px; line-height: 1.4; color: #777777; overflow-wrap: anywhere; }
.stream-control { flex: 0 0 auto; display: flex; align-items: center; gap: 5px; font-size: 12px; }
.state-bar, .error-bar, .identity-bar { flex: 0 0 auto; padding: 10px 16px; font-size: 13px; line-height: 1.6; overflow-wrap: anywhere; border-bottom: 1px solid #ececec; }
.state-bar { color: #666666; background: #f8f8f8; }
.error-bar { color: #b72a22; background: #fff5f4; }
.request-id { display: block; color: #777777; font-size: 11px; margin-top: 3px; }
.identity-bar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; background: #fafafa; }
.identity-bar text { flex: 1 1 160px; }
.login-button, .identity-button { flex: 0 0 auto; margin: 0; padding: 7px 12px; min-height: 36px; color: #ffffff; background: #dd352b; font-size: 13px; line-height: 1.5; border-radius: 4px; }
.chat-scroll { flex: 1 1 0; min-height: 0; height: 0; width: 100%; }
.chat-content { box-sizing: border-box; max-width: 1120px; margin: 0 auto; padding: 18px 16px 8px; }
.empty-chat { display: flex; gap: 12px; align-items: center; padding: 14px 0 28px; color: #666666; font-size: 14px; line-height: 1.6; }
.message-row { display: flex; margin-bottom: 14px; }
.message-row.user { justify-content: flex-end; }
.message-text { max-width: 90%; min-width: 0; padding: 10px 12px; background: #f4f4f4; border-radius: 6px; }
.user .message-text { background: #fff1f0; }
.message-role { display: block; font-size: 11px; color: #888888; line-height: 1.5; margin-bottom: 4px; }
.message-content { display: block; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; font-size: 14px; line-height: 1.75; }
.product-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 16px 0; }
.chat-end { height: 12px; }
.agent-composer { flex: 0 0 auto; border-top: 1px solid #e5e5e5; padding: 10px max(16px, env(safe-area-inset-right)) calc(9px + env(safe-area-inset-bottom)) max(16px, env(safe-area-inset-left)); background: #ffffff; }
.composer-row { max-width: 1088px; margin: 0 auto; display: flex; align-items: stretch; gap: 10px; }
.message-input { flex: 1 1 0; width: 0; min-width: 0; height: 64px; min-height: 0; max-height: 64px; box-sizing: border-box; padding: 9px 11px; border: 1px solid #dcdcdc; border-radius: 4px; color: #333333; background: #ffffff; font-size: 14px; line-height: 1.55; }
.send-button { flex: 0 0 48px; width: 48px; height: 64px; margin: 0; padding: 0; display: flex; align-items: center; justify-content: center; color: #ffffff; background: #dd352b; border-radius: 4px; }
.message-count { display: block; max-width: 1088px; margin: 4px auto 0; color: #888888; font-size: 10px; line-height: 1.4; }
.invalid-count { color: #b72a22; }
.startup-error { padding: 24px; color: #b72a22; font-size: 14px; line-height: 1.7; overflow-wrap: anywhere; }
button::after { border: 0; }
button[disabled] { opacity: 0.45; }
@media (max-width: 960px) { .product-grid { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 420px) {
  .agent-header { min-height: 62px; gap: 9px; padding-left: 12px; padding-right: 12px; }
  .chat-content { padding-left: 12px; padding-right: 12px; }
  .agent-composer { padding-left: 12px; padding-right: 12px; }
  .message-text { max-width: 95%; }
}
</style>

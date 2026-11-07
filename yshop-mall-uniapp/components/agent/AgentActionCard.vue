<template>
  <view class="agent-action" :class="{ 'action-success': result && result.status === 'succeeded' }">
    <view class="action-title">
      <uni-icons :type="result && result.status === 'succeeded' ? 'checkbox-filled' : 'cart'" :size="20" :color="result && result.status === 'succeeded' ? '#21854c' : '#d62f26'" />
      <text>{{ heading }}</text>
    </view>
    <view v-if="card" class="action-summary">
      <image v-if="card.image && !imageFailed" class="action-image" :src="card.image" :alt="card.product_name" mode="aspectFit" @error="imageFailed = true" />
      <view v-else class="action-image image-unavailable" role="img" :aria-label="card.product_name + '，图片暂不可用'">
        <uni-icons type="image" :size="25" color="#999999" />
        <text>图片暂不可用</text>
      </view>
      <view class="summary-copy">
        <text class="product-name">{{ card.product_name }}</text>
        <text class="sku-name">{{ card.sku_name }}</text>
        <view class="summary-values">
          <text>数量 {{ card.quantity }}</text>
          <text>单价 ¥{{ card.unit_price }}</text>
          <text class="total">合计 ¥{{ card.total_price }}</text>
        </view>
      </view>
    </view>
    <text v-if="card" class="expiry">{{ expired ? '确认已过期' : '有效至 ' + expiryText }}</text>
    <text v-if="recoveryOnly" class="action-warning">结果尚未确认，请勿重复提交。</text>
    <text v-if="result && result.error_code" class="action-warning">{{ resultError }}</text>
    <text v-if="!card && !result" class="action-warning">此动作尚未核验。</text>
    <view class="action-buttons">
      <template v-if="canDecide && !expired">
        <button class="confirm-button" role="button" aria-label="确认加入购物车" :disabled="busy" @click="$emit('decision', 'confirm')">
          <uni-icons type="checkmarkempty" :size="19" color="#ffffff" />
          <text>确认加入购物车</text>
        </button>
        <button class="cancel-button" role="button" aria-label="取消加购" :disabled="busy" @click="$emit('decision', 'cancel')">
          <uni-icons type="closeempty" :size="19" color="#444444" />
          <text>取消</text>
        </button>
      </template>
      <button v-if="needsRead" class="read-button" role="button" aria-label="读取动作状态" :disabled="busy" @click="$emit('recover')">
        <uni-icons type="reload" :size="19" color="#444444" />
        <text>读取动作状态</text>
      </button>
      <button v-if="result && ['succeeded', 'unknown', 'executing', 'failed'].includes(result.status)" class="cart-button" role="button" aria-label="查看商城购物车" @click="$emit('cart')">
        <uni-icons type="cart" :size="19" color="#21854c" />
        <text>查看购物车</text>
      </button>
    </view>
  </view>
</template>

<script>
import UniIcons from '@/components/uni-icons/uni-icons.vue'
import { AgentClientError } from '@/utils/agent/client.mjs'

const HEADINGS = { executing: '加购正在处理', succeeded: '已加入购物车', failed: '加购未完成', unknown: '加购结果待核验', cancelled: '已取消加购' }

export default {
  name: 'AgentActionCard',
  components: { UniIcons },
  props: {
    card: { type: Object, default: null },
    result: { type: Object, default: null },
    actionId: { type: String, default: null },
    verified: { type: Boolean, default: false },
    canDecide: { type: Boolean, default: false },
    recoveryOnly: { type: Boolean, default: false },
    busy: { type: Boolean, default: false },
    nowValue: { type: Number, required: true },
  },
  data() { return { imageFailed: false } },
  computed: {
    heading() { return this.result ? HEADINGS[this.result.status] : '待确认加购' },
    expired() { return this.card !== null && this.card.expires_at * 1000 <= this.nowValue },
    expiryText() { return this.card ? new Date(this.card.expires_at * 1000).toLocaleString('zh-CN') : '' },
    needsRead() { return this.actionId && (!this.verified || this.recoveryOnly || this.expired) },
    resultError() { return this.result && this.result.error_code ? new AgentClientError(this.result.error_code).message : '' },
  },
  watch: {
    actionId() { this.imageFailed = false },
  },
}
</script>

<style scoped>
.agent-action { padding: 16px; border: 1px solid #e5b7b4; border-left: 3px solid #d62f26; background: #ffffff; border-radius: 6px; }
.action-success { border-color: #b5d8c2; border-left-color: #21854c; }
.action-title { display: flex; align-items: center; gap: 8px; font-size: 16px; line-height: 1.5; font-weight: 600; }
.action-summary { display: flex; gap: 12px; margin-top: 12px; align-items: flex-start; }
.action-image { flex: 0 0 80px; width: 80px; height: 80px; background: #f6f6f6; border-radius: 4px; }
.image-unavailable { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px; font-size: 10px; color: #777777; }
.summary-copy { min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.product-name { font-size: 15px; line-height: 1.45; font-weight: 600; overflow-wrap: anywhere; }
.sku-name { color: #666666; font-size: 13px; overflow-wrap: anywhere; }
.summary-values { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 13px; line-height: 1.5; }
.total { color: #d62f26; font-weight: 600; overflow-wrap: anywhere; }
.expiry, .action-warning { display: block; margin-top: 10px; font-size: 12px; line-height: 1.6; color: #777777; overflow-wrap: anywhere; }
.action-warning { color: #a34816; }
.action-buttons { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.action-buttons button { min-height: 40px; margin: 0; padding: 9px 12px; display: flex; align-items: center; justify-content: center; gap: 5px; border-radius: 4px; font-size: 13px; line-height: 1.4; white-space: normal; }
.confirm-button { color: #ffffff; background: #dd352b; }
.cancel-button, .read-button { color: #333333; background: #f4f4f4; }
.cart-button { color: #21854c; background: #edf7f1; }
button::after { border: 0; }
button[disabled] { opacity: 0.45; }
@media (max-width: 420px) { .confirm-button { flex: 1 1 180px; } }
</style>

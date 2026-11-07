<template>
  <view class="agent-product">
    <view class="product-heading">
      <image v-if="picture && !imageFailed" class="product-image" :src="picture" :alt="product.name" mode="aspectFit" @error="imageFailed = true" />
      <view v-else class="product-image image-unavailable" role="img" :aria-label="product.name + '，图片暂不可用'">
        <uni-icons type="image" :size="28" color="#999999" />
        <text>图片暂不可用</text>
      </view>
      <view class="product-copy">
        <text class="product-name">{{ product.name }}</text>
        <text class="price">¥{{ priceText }}</text>
        <text class="stock">{{ selectedSku ? '库存 ' + selectedSku.stock : '库存待选规格' }}</text>
      </view>
    </view>
    <view class="product-options">
      <text class="field-label">规格</text>
      <picker class="sku-picker" mode="selector" :range="product.skus" range-key="name" :value="selectedIndex" :disabled="disabled || !product.skus.length" role="button" :aria-label="product.name + '的规格'" @change="selectSku">
        <view class="picker-value">
          <text>{{ selectedSku ? selectedSku.name : '请选择规格' }}</text>
          <uni-icons type="arrowdown" :size="16" color="#666666" />
        </view>
      </picker>
      <text class="field-label">数量</text>
      <view class="quantity-control">
        <button class="step-button" role="button" aria-label="减少数量" title="减少数量" :disabled="disabled || quantity === null || quantity <= 1" @click="stepQuantity(-1)">
          <uni-icons type="minus" :size="18" color="#444444" />
        </button>
        <input v-model="quantityText" class="quantity-input" type="number" :disabled="disabled" :maxlength="5" :aria-label="product.name + '的数量'" />
        <button class="step-button" role="button" aria-label="增加数量" title="增加数量" :disabled="disabled || quantity === null || quantity >= 9999" @click="stepQuantity(1)">
          <uni-icons type="plus" :size="18" color="#444444" />
        </button>
      </view>
    </view>
    <view class="product-commands">
      <button class="detail-button" role="button" :aria-label="'查看' + product.name + '详情'" @click="$emit('open', product.product_id)">
        <uni-icons type="arrowright" :size="17" color="#444444" />
        <text>商品详情</text>
      </button>
      <button class="prepare-button" role="button" :aria-label="'准备加购' + product.name" :disabled="!canPrepare" @click="prepare">
        <uni-icons type="cart" :size="18" color="#ffffff" />
        <text>准备加购</text>
      </button>
    </view>
  </view>
</template>

<script>
import UniIcons from '@/components/uni-icons/uni-icons.vue'

export default {
  name: 'AgentProductCard',
  components: { UniIcons },
  props: {
    product: { type: Object, required: true },
    revision: { type: Number, required: true },
    disabled: { type: Boolean, default: false },
  },
  data() {
    return {
      selectedId: this.product.skus.length === 1 ? this.product.skus[0].unique_id : null,
      quantityText: '1',
      imageFailed: false,
    }
  },
  computed: {
    selectedSku() {
      return this.product.skus.find(item => item.unique_id === this.selectedId) || null
    },
    selectedIndex() {
      return Math.max(0, this.product.skus.findIndex(item => item.unique_id === this.selectedId))
    },
    picture() {
      return this.selectedSku && this.selectedSku.image || this.product.image
    },
    priceText() {
      if (this.selectedSku) return this.selectedSku.price
      const lowest = this.product.skus.slice().sort((a, b) => Number(a.price) - Number(b.price))[0]
      return lowest ? lowest.price + ' 起' : '--'
    },
    quantity() {
      if (!/^[0-9]{1,4}$/.test(this.quantityText)) return null
      const quantity = Number(this.quantityText)
      return Number.isSafeInteger(quantity) && quantity >= 1 && quantity <= 9999 ? quantity : null
    },
    canPrepare() {
      return !this.disabled && this.selectedSku !== null && this.quantity !== null && this.quantity <= this.selectedSku.stock
    },
  },
  watch: {
    picture() { this.imageFailed = false },
  },
  methods: {
    selectSku(event) {
      const index = Number(event.detail.value)
      if (Number.isSafeInteger(index) && index >= 0 && index < this.product.skus.length) this.selectedId = this.product.skus[index].unique_id
    },
    stepQuantity(delta) {
      if (this.quantity !== null) this.quantityText = String(Math.min(9999, Math.max(1, this.quantity + delta)))
    },
    prepare() {
      if (!this.canPrepare) return
      this.$emit('prepare', { productId: this.product.product_id, uniqueId: this.selectedSku.unique_id, quantity: this.quantity, revision: this.revision })
    },
  },
}
</script>

<style scoped>
.agent-product { min-width: 0; padding: 14px; background: #ffffff; border: 1px solid #e4e4e4; border-radius: 6px; }
.product-heading { display: flex; gap: 12px; align-items: flex-start; }
.product-image { flex: 0 0 86px; width: 86px; height: 86px; background: #f6f6f6; border-radius: 4px; }
.image-unavailable { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px; color: #777777; font-size: 10px; }
.product-copy { min-width: 0; display: flex; flex-direction: column; gap: 7px; }
.product-name { font-size: 16px; line-height: 1.4; font-weight: 600; overflow-wrap: anywhere; }
.price { color: #d62f26; font-size: 17px; font-weight: 600; overflow-wrap: anywhere; }
.stock { color: #777777; font-size: 12px; }
.product-options { display: grid; grid-template-columns: 36px minmax(0, 1fr); gap: 10px 8px; align-items: center; margin-top: 15px; }
.field-label { font-size: 13px; color: #666666; }
.sku-picker { min-width: 0; border: 1px solid #dddddd; border-radius: 4px; }
.picker-value { min-height: 36px; padding: 7px 10px; box-sizing: border-box; display: flex; align-items: center; justify-content: space-between; gap: 8px; font-size: 13px; }
.picker-value text { min-width: 0; overflow-wrap: anywhere; }
.quantity-control { width: 136px; height: 36px; display: grid; grid-template-columns: 36px 64px 36px; border: 1px solid #dddddd; border-radius: 4px; overflow: hidden; box-sizing: content-box; }
.step-button { width: 36px; height: 36px; margin: 0; padding: 0; display: flex; align-items: center; justify-content: center; background: #f7f7f7; border-radius: 0; line-height: 1; }
.quantity-input { width: 64px; height: 36px; min-height: 0; text-align: center; font-size: 14px; }
.product-commands { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 8px; margin-top: 14px; }
.product-commands button { min-width: 0; min-height: 38px; margin: 0; padding: 8px; display: flex; justify-content: center; align-items: center; gap: 5px; font-size: 13px; line-height: 1.4; border-radius: 4px; white-space: normal; }
.detail-button { background: #f5f5f5; color: #333333; }
.prepare-button { background: #dd352b; color: #ffffff; }
button::after { border: 0; }
button[disabled] { opacity: 0.45; }
</style>

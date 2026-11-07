<template>
  <view v-if="sources.length || notice" class="agent-sources">
    <text class="sources-heading">规则依据</text>
    <view v-for="(source, index) in sources" :key="source.chunk_id + ':' + index" class="source-item">
      <text class="source-title">{{ source.title }}</text>
      <text class="source-section">{{ source.section }}</text>
      <text class="source-text">{{ source.text }}</text>
      <text class="source-meta">演示资料 · {{ source.version }}</text>
    </view>
    <text v-if="notice" class="source-notice">{{ notice }}</text>
  </view>
</template>

<script>
export default {
  name: 'AgentSources',
  props: {
    sources: { type: Array, default: () => [] },
    notice: { type: String, default: null },
  },
}
</script>

<style scoped>
.agent-sources { padding-top: 14px; border-top: 1px solid #e5e5e5; }
.sources-heading { font-size: 14px; font-weight: 600; color: #444444; }
.source-item { padding: 12px 0; border-bottom: 1px solid #ededed; }
.source-title, .source-section, .source-text, .source-meta, .source-notice { display: block; overflow-wrap: anywhere; }
.source-title { font-size: 14px; line-height: 1.5; font-weight: 600; }
.source-section { color: #666666; font-size: 12px; line-height: 1.6; margin-top: 3px; }
.source-text { margin-top: 7px; color: #444444; font-size: 13px; line-height: 1.7; white-space: pre-wrap; }
.source-meta, .source-notice { margin-top: 8px; color: #777777; font-size: 12px; line-height: 1.6; }
</style>

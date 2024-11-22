<template>
  <v-skeleton-loader
    :loading="estimationStore.networkGraphLoading"
    type="image"
    class="justify-center"
  >
    <VueMermaidString :value="networkGraph" :options="{ theme: 'neutral' }" @click="handleSvgClick" />
  </v-skeleton-loader>
</template>

<script setup>
import VueMermaidString from 'vue-mermaid-string'

const estimationStore = useEstimationStore()

const networkGraph = computed(() => estimationStore.networkGraphGetter)

const handleSvgClick = (event) => {
  // This is only necessary, because a bug in mermaidjs is breaking the @node-click functionality of vue-mermaid-string.
  // See also: https://github.com/dword-design/vue-mermaid-string/issues/197 and https://github.com/mermaid-js/mermaid/issues/4346
  const target = event.target
  const group = target.closest('g')
  if (group && group.id) {
    estimationStore.selectedNode = group.id.split('-')[1]
  }
}
</script>

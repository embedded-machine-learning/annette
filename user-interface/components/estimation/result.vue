<template>
    <v-skeleton-loader
        :loading="estimationStore.loading"    
        type="image"
        class="d-flex justify-center"
    >
        <span
            class="text-h1"
        >{{ formattedEstimation }} ms</span>
    </v-skeleton-loader>
</template>

<script setup>
const estimationStore = useEstimationStore()

const formattedEstimation = computed(() => {
    return estimationStore.estimation ? estimationStore.estimation.toFixed(4) : null
})
</script>

<template>
    <v-skeleton-loader type="table" :loading="estimationStore.loading">
        <v-data-table
        :headers="headers"
        :items="formattedLayerResult"
        items-per-page="25"
        density="compact"
        />
    </v-skeleton-loader>
</template>

<script setup>
const estimationStore = useEstimationStore()

const headers = [
    {
        title: "Index",
        value: "index"
    },
    {
        title: "Name",
        value: "name"
    },
    {
        title: "Type",
        value: "type"
    },
    {
        title: "Time (ms)",
        value: "time(ms)"
    },
    {
        title: "Number of Operations",
        value: "num_ops"
    },
    {
        title: "Number of Inputs",
        value: "num_inputs"
    },
    {
        title: "Number of Outputs",
        value: "num_outputs"
    },
    {
        title: "Number of Weights",
        value: "num_weights"
    }
]

const formattedLayerResult = computed(() => {
    const layerResult = estimationStore.layerResult
    if (!layerResult) {
      return []
    }
    var formattedOutput = layerResult.map((layer, i) => {
      return {
        ...layer,
        'time(ms)': parseFloat(layer['time(ms)']),
        index: i
      }
    })
    return formattedOutput
})
</script>
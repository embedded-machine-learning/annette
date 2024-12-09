<template>
    <v-skeleton-loader type="table" :loading="estimationStore.loading">
        <v-data-table
        :headers="headers"
        :items="formattedLayerResult"
        items-per-page="25"
        density="compact"
        >
            <template v-slot:item="{ item }">
                <tr @click="clickedRow(item)" :class="{'selected-row': item.name === estimationStore.selectedNode}" style="cursor: pointer;">
                    <td>{{ item.index }}</td>
                    <td>{{ item.name }}</td>
                    <td>{{ item.type }}</td>
                    <td>{{ item['time(ms)'] }}</td>
                    <td>{{ item.num_ops }}</td>
                    <td>{{ item.num_inputs }}</td>
                    <td>{{ item.num_outputs }}</td>
                    <td>{{ item.num_weights }}</td>
                </tr>
            </template>
        </v-data-table>
    </v-skeleton-loader>
</template>

<script setup>
const estimationStore = useEstimationStore()

const headers = [
    {
        title: "Index",
        value: "index",
        sortable: true
    },
    {
        title: "Name",
        value: "name",
        sortable: true
    },
    {
        title: "Type",
        value: "type",
        sortable: true
    },
    {
        title: "Time (ms)",
        value: "time(ms)",
        sortable: true
    },
    {
        title: "Number of Operations",
        value: "num_ops",
        sortable: true
    },
    {
        title: "Number of Inputs",
        value: "num_inputs",
        sortable: true
    },
    {
        title: "Number of Outputs",
        value: "num_outputs",
        sortable: true
    },
    {
        title: "Number of Weights",
        value: "num_weights",
        sortable: true
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

const clickedRow = (item) => {
    if (estimationStore.selectedNode !== item.name) {
        estimationStore.selectedNode = item.name
    } else {
        estimationStore.selectedNode = null
    }
}
</script>

<style scoped>
.selected-row {
    background-color: #5188c2;
}
</style>
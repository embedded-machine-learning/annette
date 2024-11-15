<template>
    <v-row>
        <h1>Exploration</h1>
    </v-row>
    <v-row>
        <v-col
            cols="12"
            sm="10"
        >
            <v-row>
                <v-col
                    cols="12"
                    sm="5"
                >
                    <Select v-model="selectedNetwork" label="Network" :items="databaseStore.onnxNetworks" clearable />
                </v-col>
                <v-col
                    cols="12"
                    sm="2"
                    align="center"
                >
                    <span class="text-caption">
                        OR
                    </span>
                </v-col>
                <v-col
                    cols="12"
                    sm="5"
                >
                    <Select v-model="selectedLayerModel" label="Hardware Platform" :items="databaseStore.layerModelSelectorItems" itemValue="name" :itemProps="layerModelItemProps" clearable />
                </v-col>
            </v-row>
        </v-col>
        <v-col
            cols="12"
            sm="2"
        >
            <v-btn
                @click="runExploration"
                :loading="explorationStore.loading"
                prepend-icon="mdi-compass"
                variant="outlined"
            >
                Explore
                <template v-slot:loader>
                    <v-progress-circular indeterminate :width="4" />
                </template>
            </v-btn>
        </v-col>
    </v-row>
    <v-divider />
    <v-container v-if="explorationStore.results.length > 0" height="100%">
        <Barchart />
    </v-container>
    <v-container v-else align="center">
        <span
            class="text-overline"
        >Please select either a network or a hardware platform!</span>
    </v-container>
</template>

<script setup>
import { ref, watch } from 'vue'
import Select from '../components/forms/select.vue'
import Barchart from '../components/charts/barchart-exploration.vue'

const databaseStore = useDatabaseStore()
const explorationStore = useExplorationStore()

const selectedNetwork = ref()
const selectedLayerModel = ref()
const layerModelItemProps = (item) => {
        return {
            title: item.name,
            subtitle: item.description
        }
    }

watch(selectedNetwork, (newVal) => {
    if (newVal) {
        selectedLayerModel.value = null
    }
})

watch(selectedLayerModel, (newVal) => {
    if (newVal) {
        selectedNetwork.value = null
    }
})

const runExploration = async () => {
    await explorationStore.runExploration(selectedNetwork.value, selectedLayerModel.value)
}
</script>
<template>
    <v-row>
        <h1>Estimation</h1>
    </v-row>
    <v-form v-model="valid" ref="form">
        <v-row>
            <v-col
                cols="12"
                sm="10"
            >
                <v-row>
                    <v-col
                        v-for="(selector, i) in estimationSelectors" :key="i"
                        cols="12"
                        sm="4"
                    >
                        <UploadSelect :id="`UploadSelector-${i}`" v-model="selector.selectedValue" :items="selector.items" :label="selector.label" :item-value="selector.itemValue" :itemProps="selector.itemProps" :rules="validationRules" required />
                    </v-col>
                </v-row>
            </v-col>
            <v-col
                cols="12"
                sm="2"
            >
                <v-btn
                    @click="runEstimation"
                    :loading="estimationStore.loading"
                    prepend-icon="mdi-rocket-launch"
                    variant="outlined"
                >
                    Estimate
                    <template v-slot:loader>
                        <v-progress-circular indeterminate :width="4" />
                    </template>
                </v-btn>
            </v-col>
        </v-row>
    </v-form>
    <v-divider />
    <v-container v-if="estimationStore.estimation">
        <v-row>
            <v-col
                cols="12"
                sm="8"
            >
                <v-row>
                    <v-col>
                        <v-row
                            justify="center"
                        >
                            <Result />
                        </v-row>
                        <v-row
                            class="my-3"
                            justify="center"
                        >
                            <v-skeleton-loader
                                :loading="estimationStore.loading"    
                                type="text"
                                class="d-flex justify-center"
                            >
                                <span>Network: {{ estimationStore.network }} | Hardware Platform: {{ estimationStore.layerModel }} | Mapping Model: {{ estimationStore.mappingModel }}</span>
                            </v-skeleton-loader>
                        </v-row>
                    </v-col>
                </v-row>
                <v-row>
                    <LayerTable />
                </v-row>
            </v-col>
            <v-col
                cols="12"
                sm="4"
            >
                <v-col>
                    <NetworkGraph />
                </v-col>
            </v-col>
        </v-row>
    </v-container>
    <v-container v-else align="center">
        <span
            class="text-overline"
        >Please select a network, a hardware platform and a mapping model to run an estimation!</span>
    </v-container>
</template>

<script setup>
import { reactive, ref } from 'vue'
import UploadSelect from '../components/forms/upload-select.vue'
import Result from '../components/estimation/result.vue'
import LayerTable from '../components/estimation/layer-table.vue'
import NetworkGraph from '../components/estimation/network-graph.vue'
import { useEstimationStore } from '~/stores/estimation'
import { parseFilename } from '../utils/index'

const estimationStore = useEstimationStore()
const databaseStore = useDatabaseStore()

const form = ref('form')
const valid = ref(false)

const estimationSelectors = reactive({
    network: {
        label: "Network",
        items: databaseStore.onnxNetworks,
        selectedValue: undefined
    },
    layerModel: {
        label: "Hardware Platform",
        items: databaseStore.layerModelSelectorItems,
        selectedValue: undefined,
        itemValue: 'name',
        itemProps (item) {
            return {
                title: item.name,
                subtitle: item.description
            }
        }
    },
    mappingModel: {
        label: "Mapping Model",
        items: databaseStore.mappingModels,
        selectedValue: undefined
    }
})

const validationRules = ref([
    value => {
        if (value) return true
        return 'Please make a selection'
    }
])

const runEstimation = async () => {
    form.value.validate()
    if (!valid.value) {
        console.warn('Validation of the form failed.')
        return
    }
    const network = parseFilename(estimationSelectors.network.selectedValue)
    const layerModel = parseFilename(estimationSelectors.layerModel.selectedValue)
    const mappingModel = parseFilename(estimationSelectors.mappingModel.selectedValue)
    await estimationStore.runEstimation(network, layerModel, mappingModel)
}
</script>
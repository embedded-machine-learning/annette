export const useEstimationStore = defineStore('estimationStore', {
    state: () => ({
      loading: false,
      network: null,
      layerModel: null,
      mappingModel: null,
      estimation: null,
      layerResult: []
    }),
    actions: {
      async runEstimation (network, layerModel, mappingModel) {
        this.loading = true
        const BACKEND_URL = useRuntimeConfig().public.backendURL
        const { estimation, layerResult } = await $fetch(`http://${BACKEND_URL}/estimate?network=${network}&layer=${layerModel}&mapping=${mappingModel}`)
        this.network = network
        this.layerModel = layerModel
        this.mappingModel = mappingModel
        this.estimation = estimation
        this.layerResult = layerResult
        this.loading = false
      }
    }
  })
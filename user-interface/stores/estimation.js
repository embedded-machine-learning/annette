export const useEstimationStore = defineStore('estimationStore', {
    state: () => ({
      loading: false,
      networkGraphLoading: false,
      network: null,
      layerModel: null,
      mappingModel: null,
      estimation: null,
      layerResult: [],
      networkGraph: '',
      selectedNode: null
    }),
    getters: {
      networkGraphGetter: (state) => {
        if (!state.selectedNode) {
          return state.networkGraph
        } else {
          return state.networkGraph + `\nstyle ${state.selectedNode} fill:#0856a8b3,stroke:#333,stroke-width:2px`
        }
      },
    },
    actions: {
      async runEstimation (network, layerModel, mappingModel) {
        this.loading = true
        this.networkGraphLoading = true
        const BACKEND_URL = useRuntimeConfig().public.backendURL
        const { estimation, layerResult } = await $fetch(`http://${BACKEND_URL}/estimate?network=${network}&layer=${layerModel}&mapping=${mappingModel}`)
        this.network = network
        this.layerModel = layerModel
        this.mappingModel = mappingModel
        this.estimation = estimation
        this.layerResult = layerResult
        this.getNetworkGraph()
        this.loading = false
      },
      async getNetworkGraph () {
        this.networkGraphLoading = true
        this.selectedNode = null
        const BACKEND_URL = useRuntimeConfig().public.backendURL
        this.networkGraph = await $fetch(`http://${BACKEND_URL}/network-graph?network=${this.network}`)
        this.networkGraphLoading = false
      }
    }
  })
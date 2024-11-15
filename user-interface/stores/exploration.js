import { parseFilename } from '../utils/index'

export const useExplorationStore = defineStore('explorationStore', {
    state: () => ({
      loading: false,
      network: null,
      layerModel: null,
      results: []
    }),
    actions: {
      async runExploration (network, layerModel) {
        this.loading = true
        const BACKEND_URL = useRuntimeConfig().public.backendURL
        try {
          if (network && !layerModel) {
            this.results = await $fetch(`http://${BACKEND_URL}/results?network=${parseFilename(network)}`)
          } else if (!network && layerModel) {
            this.results = await $fetch(`http://${BACKEND_URL}/results?layer=${parseFilename(layerModel)}`)
          } else if (!network && !layerModel) {
            this.results = await $fetch(`http://${BACKEND_URL}/results`)
          } else {
            console.log('Setting both, the network and the layerModel is not supported.')
          }
        } catch (e) {
          console.error('Result could not be fetched...')
        }
        this.network = network
        this.layerModel = layerModel
        this.loading = false
      }
    }
})
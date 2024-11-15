export const useLayerAnalysisStore = defineStore('layerAnalysisStore', {
  state: () => ({
    loading: false,
    layerAnalysis: {}
  }),
  actions: {
    async initialize () {
      this.loading = true
      const BACKEND_URL = useRuntimeConfig().public.backendURL
      try {
          this.layerAnalysis = await $fetch(`http://${BACKEND_URL}/layer-analysis`)
      } catch (error) {
          console.error('Layer-Analysis was not able to be initialized...')
      }
    }
  }
})
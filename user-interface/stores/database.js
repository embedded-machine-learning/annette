export const useDatabaseStore = defineStore('databaseStore', {
    state: () => ({
      loading: false,
      uploadLoading: false,
      onnxNetworks: [],
      layerModels: [],
      mappingModels: []
    }),
    getters: {
      layerModelSelectorItems: (state) => state.layerModels.map(model => {
        const { hardware_description } = model

        const subtitle = [
          hardware_description?.name ? `Name: ${hardware_description.name}` : null,
          hardware_description?.hardware_type ? `Type: ${hardware_description.hardware_type}` : null,
          hardware_description?.price ? `Price: ${hardware_description.price}€` : null,
        ].filter(Boolean).join(' | ')

        return {
          name: model.name,
          description: subtitle || undefined,
        }
      })
    },
    actions: {
      async initialize () {
        this.loading = true
        const BACKEND_URL = useRuntimeConfig().public.backendURL
        try {
            this.onnxNetworks = await $fetch(`http://${BACKEND_URL}/networks`)
            this.layerModels = await $fetch(`http://${BACKEND_URL}/layer-models`)
            this.mappingModels = await $fetch(`http://${BACKEND_URL}/mapping-models`)
        } catch (error) {
            console.error('Database was not able to be initialized...')
        }
      }
    }
  })
// https://nuxt.com/docs/api/configuration/nuxt-config
import vuetify, { transformAssetUrls } from 'vite-plugin-vuetify'

export default defineNuxtConfig({
  build: {
    transpile: ['vuetify'],
  },
  modules: [
    'nuxt-security',
    '@pinia/nuxt',
    (_options, nuxt) => {
      nuxt.hooks.hook('vite:extendConfig', (config) => {
        config.plugins.push(vuetify({ autoImport: true }))
      })
    }
  ],
  vite: {
    vue: {
      template: {
        transformAssetUrls
      }
    }
  },
  security: {
    requestSizeLimiter: {
      maxRequestSizeInBytes: 50 * 1024 * 1024,
      maxUploadFileRequestInBytes: 50 * 1024 * 1024,
      throwError: false
    }
  },
  compatibilityDate: '2024-04-03',
  devtools: { enabled: true },
  runtimeConfig: {
    public: {
      backendURL: 'localhost:5000'
    }
  }
})

<template>
  <v-row
    justify="space-between"
  >
    <v-col
      cols="auto"
    >
      <h1>Database</h1>
    </v-col>
    <v-col
      cols="auto"
    >
      <UploadButton />
    </v-col>
  </v-row>
  <v-container>
    <v-tabs
      v-model="selectedTab"
      align-tabs="center"
    >
      <v-tab v-for="tab in tabs" :value="tab.value">
        <v-icon :icon="tab.icon" />
        {{ tab.title }}
      </v-tab>
    </v-tabs>
    <client-only>
      <v-tabs-window v-model="selectedTab">
        <v-tabs-window-item
          value="networks"
        >
          <v-data-table
            :headers="headers.networks"
            :items="transformedNetworks"
            density="compact"
          ></v-data-table>
        </v-tabs-window-item>
        <v-tabs-window-item
          value="hardware"
        >
          <v-data-table
            :headers="headers.hardware"
            :items="databaseStore.layerModels"
            density="compact"
          ></v-data-table>
        </v-tabs-window-item>
        <v-tabs-window-item
          value="mappings"
        >
          <v-data-table
            :headers="headers.mappings"
            :items="transformedMappingModels"
            density="compact"
          ></v-data-table>
        </v-tabs-window-item>
      </v-tabs-window>
    </client-only>
  </v-container>
</template>

<script setup>
import UploadButton from '../components/forms/upload-button.vue'

const databaseStore = useDatabaseStore()

const selectedTab = ref('networks')

const tabs = ref([
  {
    'value': 'networks',
    'icon': 'mdi-graph',
    'title': 'Networks'
  },
  {
    'value': 'hardware',
    'icon': 'mdi-cpu-64-bit',
    'title': 'Hardware'
  },
  {
    'value': 'mappings',
    'icon': 'mdi-swap-horizontal',
    'title': 'Mappings'
  }
])

const headers = ref({
  networks: [
    {
      title: 'Network',
      key: 'name'
    }
  ],
  hardware: [
    {
      title: 'Hardware Platforms',
      key: 'name'
    },
    {
      title: 'Hardware Name',
      key: 'hardware_description.name'
    },
    {
      title: 'Number of Cores',
      key: 'hardware_description.hardware_settings.num_cores',
      align: 'center'
    },
    {
      title: 'Core Frequency [GHz]',
      key: 'hardware_description.hardware_settings.core_frequency',
      align: 'center'
    },
    {
      title: 'Memory Frequency [GHz]',
      key: 'hardware_description.hardware_settings.memory_frequency',
      align: 'center'
    },
    {
      title: 'Memory Bandwidth [GB/s]',
      key: 'hardware_description.hardware_settings.memory_bandwidth',
      align: 'center'
    },
    {
      title: 'Memory Size [Gb]',
      key: 'hardware_description.hardware_settings.memory_size',
      align: 'center'
    },
    {
      title: 'Price [€]',
      key: 'hardware_description.price',
      align: 'center'
    }
  ],
  mappings: [
    {
      title: 'Mapping Models',
      key: 'name'
    }
  ]
})

const transformedNetworks = computed(() => {
  return databaseStore.onnxNetworks.map(network => {
    return {
      name: network
    }
  })
})

const transformedMappingModels = computed(() => {
  return databaseStore.mappingModels.map(model => {
    return {
      name: model
    }
  })
})
</script>
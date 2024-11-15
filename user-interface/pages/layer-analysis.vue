<template>
    <h1>Layer Analysis</h1>
    <v-row>
        <v-col
          cols="12"
          sm="6"
        >
          <Barchart :data="layerAnalysisStore.layerAnalysis" attribute="average" title="Average Execution Time (ms)" :chartOptions="chartOptionsAverage" />
        </v-col>
        <v-col
          cols="12"
          sm="6"
        >
          <Barchart :data="layerAnalysisStore.layerAnalysis" attribute="count" title="Number of times included in Results" :chartOptions="chartOptionsCount" />
        </v-col>
    </v-row>
    <v-row>
      <v-col
        cols="12"
        sm="12"
      >
        <ScatterChart :data="layerAnalysisStore.layerAnalysis" />
      </v-col>
    </v-row>
</template>

<script setup>
import Barchart from '../components/charts/barchart-layer-analysis.vue'
import ScatterChart from '../components/charts/scatter-layer-analysis.vue'

const layerAnalysisStore = useLayerAnalysisStore()

layerAnalysisStore.initialize()

const chartOptionsAverage = ref({
  indexAxis: 'y',
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    x: {
      text: 'Time (ms)',
      type: 'logarithmic',
      ticks: {
        callback: function (value, index, ticks) {
          return value + ' ms'
        }
      }
    }
  },
  plugins: {
    tooltip: {
      callbacks: {
        label: function (TooltipItem) {
          return TooltipItem.dataset.label + ': ' + TooltipItem.raw + ' ms'
        }
      }
    }
  }
})

const chartOptionsCount = ref({
  indexAxis: 'y',
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    x: {
      text: 'Time (ms)',
      type: 'logarithmic',
      ticks: {
        callback: (value) => {
          if (value % 1 === 0) {
            return value
          }
        }
      }
    }
  },
  plugins: {
    tooltip: {
      callbacks: {
        label: function (TooltipItem) {
            return TooltipItem.dataset.label + ': ' + TooltipItem.formattedValue + ' layers'
        }
      }
    }
  }
})
</script>
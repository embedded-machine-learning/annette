<template>
  <div style="height: 80vh;">
    <Bar
      :data="chartData"
      :options="chartOptions"
    />
  </div>
</template>

<script setup>
import { Bar } from 'vue-chartjs'
import { getChartColor } from '../../utils/index'

const explorationStore = useExplorationStore()

const chartData = computed(() => {
  const network = explorationStore.network
  const layerModel = explorationStore.layerModel
  if (network || layerModel) {
    return chartDataSpecific.value
  } else {
    return chartDataGeneral.value
  }
})

const chartDataSpecific = computed(() => {
  const network = explorationStore.network
  const layerModel = explorationStore.layerModel
  const results = explorationStore.results

  var title = ''
  var labels = []
  if (network && !layerModel) {
    title = network
    results.sort(function(a, b) {
      var hardwareA = a.hardware.toLowerCase()
      var hardwareB = b.hardware.toLowerCase()
      return (hardwareA < hardwareB) ? -1 : (hardwareA > hardwareB) ? 1 : 0
    })
    labels = results.map(r => r['hardware'])
  } else if (!network && layerModel) {
    title = layerModel
    results.sort(function(a, b) {
      var networkA = a.network.toLowerCase()
      var networkB = b.network.toLowerCase()
      return (networkA < networkB) ? -1 : (networkA > networkB) ? 1 : 0
    })
    labels = results.map(r => r['network'])
  }

  return {
    labels: labels,
    datasets: [
      {
        label: title,
        backgroundColor: getChartColor(0),
        data: results.map(r => r['sum']),
      }
    ]
  }
})

const chartDataGeneral = computed(() => {
  const results = explorationStore.results
  results.sort(function(a, b) {
    var networkA = a.network.toLowerCase()
    var networkB = b.network.toLowerCase()
    return (networkA < networkB) ? -1 : (networkA > networkB) ? 1 : 0
  })
  var networks = [...new Set(results.map(r => r['network']))]
  var hardwarePlatforms = [...new Set(results.map(r => r['hardware']))]

  var datasets = hardwarePlatforms.map((hardware, i) => {
    var affectedResults = results.filter(r => r['hardware'] === hardware)
    return {
      label: hardware,
      backgroundColor: getChartColor(i),
      data: affectedResults.map(r => r['sum'])
    }
  })

  return {
    labels: networks,
    datasets: datasets
  }
})

const chartOptions = ref({
  indexAxis: 'y',
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    x: {
      title: {
        display: true,
        text: 'Time (ms)',
      },
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
          return TooltipItem.dataset.label + ': ' + TooltipItem.formattedValue + ' ms'
        }
      }
    }
  }
})
</script>
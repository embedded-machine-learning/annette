<template>
  <Scatter
    :data="chartData"
    :options="chartOptions"
  />
</template>

<script setup>
import { Scatter } from 'vue-chartjs'
import { getChartColor } from '../../utils/index'

const props = defineProps(['data'])

const chartData = computed(() => {
  const datasets = Object.keys(props.data).map((type, i) => {
    return {
      label: type,
      data: props.data[type].details.map(d => {
        return {
          x: props.data[type]['count'],
          y: d['estimate'],
          result: d['result']
        }
      }),
      backgroundColor: getChartColor(i)
    }
  })
  return {
    datasets: datasets
  }
})

const chartOptions = {
  scales: {
    y: {
      title: {
        display: true,
        text: 'Time (ms)'
      },
      type: 'logarithmic',
      ticks: {
        callback: function (value, index, ticks) {
          return value + ' ms'
        }
      }
    },
    x: {
      title: {
        display: true,
        text: 'Occurance in Results'
      }
    }
  },
  plugins: {
    tooltip: {
      callbacks: {
        title: function (TooltipItem) {
          const layerTypes = [...new Set(TooltipItem.map(i => i.dataset.label))]
          return layerTypes.join(', ')
        },
        label: function (TooltipItem) {
          return TooltipItem.raw.result + ': ' + TooltipItem.raw.y + ' ms'
        }
      }
    }
  }
}
</script>
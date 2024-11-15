<template>
  <div style="height: 60vh;">
    <Bar
      :data="chartData"
      :options="chartOptions"
    />
  </div>
</template>

<script setup>
import { Bar } from 'vue-chartjs'
import { getChartColor } from '../../utils/index'

const props = defineProps(['chartOptions', 'data', 'attribute', 'title'])

const chartData = computed(() => {
  return {
    labels: Object.keys(props.data),
    datasets: [
      {
        label: props.title,
        backgroundColor: getChartColor(0),
        data: Object.values(props.data).map(d => d[props.attribute]),
      }
    ]
  }
})
</script>
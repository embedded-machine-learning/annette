import { Chart, Title, Tooltip, Legend, BarElement, PointElement, LineElement, CategoryScale, LinearScale, LogarithmicScale } from 'chart.js'

export default defineNuxtPlugin(() => {
    Chart.register(CategoryScale, LinearScale, LogarithmicScale, BarElement, PointElement, LineElement, Title, Tooltip, Legend)
})
const chartColors = [
  'rgba(8, 86, 168, 0.7)',
  'rgba(8, 136, 168, 0.7)',
  'rgba(28, 32, 51, 0.7)',
  'rgba(65, 115, 168, 0.7)',
  'rgba(33, 8, 168, 0.7)',
  'rgba(8, 168, 149, 0.7)',
  'rgba(28, 46, 51, 0.7)',
  'rgba(8, 35, 168, 0.7)'
]

const getChartColor = (index: number) => {
  return chartColors[index % chartColors.length]
}

const parseFilename = (filename: string) => {
  return filename.replace(/\.[^/.]+$/, "")
}

export {
  chartColors,
  getChartColor,
  parseFilename
}
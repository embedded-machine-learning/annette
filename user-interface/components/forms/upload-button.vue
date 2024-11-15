<template>
  <v-form @submit.prevent="uploadFile">
    <v-row
      align="center"
    >
      <v-col>
        <v-file-input
          @change="onFileChange"
          :messages="fileInputMessage"
          :error="fileInputErrorStatus"
          ref="fileInput"
          label="ONNX file"
          variant="outlined"
          density="compact"
          prepend-icon="mdi-graph"
          accept=".onnx"
          hide-details="auto"
          show-size
          style="min-width: 20vw;"
        />
      </v-col>
      <v-col>
        <v-btn
          :loading="uploadLoading"
          type="submit"
          icon="mdi-upload"
          variant="outlined"
          density="comfortable"
        />
      </v-col>
    </v-row>
  </v-form>
</template>

<script setup>
const databaseStore = useDatabaseStore()
const BACKEND_URL = useRuntimeConfig().public.backendURL

const selectedFile = ref()
const fileInput = ref('fileInput')
const fileInputMessage = ref()
const fileInputErrorStatus = ref(false)
const uploadLoading = ref(false)

const onFileChange = (event) => {
  selectedFile.value = event.target.files[0]
}

const isONNXFile = (file) => {
  return file.name.toLowerCase().endsWith('.onnx');
}

const uploadFile = async () => {
  if (!selectedFile.value) {
    fileInputMessage.value = 'Please provide a file!'
    fileInputErrorStatus.value = true
    return
  }
  if (!isONNXFile(selectedFile.value)) {
    fileInputMessage.value = 'Please provide an .onnx file!'
    fileInputErrorStatus.value = true
    return
  }
  const formData = new FormData()
  formData.append("file", selectedFile.value)
  try {
    uploadLoading.value = true
    const response = await fetch(`http://${BACKEND_URL}/upload`, {
      method: "POST",
      body: formData
    })
    uploadLoading.value = false
    if (response.ok) {
      fileInputMessage.value = 'Upload successful!'
      fileInputErrorStatus.value = false
      selectedFile.value = null
      fileInput.value = null
      databaseStore.initialize()
    } else {
      if (response.status === 409) {
        fileInputMessage.value = 'A file with this name already exists.'
        fileInputErrorStatus.value = true
      } else {
        fileInputMessage.value = 'Invalid file. Please provide an .onnx file!'
        fileInputErrorStatus.value = true
      }
    }
  } catch (error) {
    fileInputMessage.value = 'An error occured while uploading the file...'
    fileInputErrorStatus.value = true
  }
}
</script>

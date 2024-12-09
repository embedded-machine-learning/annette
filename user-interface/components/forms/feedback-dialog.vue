<template>
  <v-dialog
    max-width="500"
    persistent
  >
    <template v-slot:activator="{ props: activatorProps }">
      <v-list nav>
        <v-list-item link rounded="lg" v-bind="activatorProps">
          <template v-slot:prepend>
            <v-icon>mdi-email-outline</v-icon>
          </template>
          <v-list-item-title v-text="'Feedback'"></v-list-item-title>
        </v-list-item>
      </v-list>
    </template>
    <template v-slot:default="{ isActive }">
      <v-card>
        <v-card-text>
          <p class="my-2">
            Please provide your feedback in the textarea below.
          </p>
          <v-form v-model="valid" ref="form">
            <v-textarea
              v-model="feedbackInput"
              :rules="validationRules"
              variant="outlined"
              density="compact"
              counter="1000"
            />
          </v-form>
        </v-card-text>
        <v-card-actions>
          <v-btn
            text="Cancel"
            @click="isActive.value = false"
          ></v-btn>
          <v-spacer />
          <v-btn
            @click="sendFeedback"
            append-icon="mdi-send"
            variant="outlined"
          >
            Send
            <template v-slot:loader>
              <v-progress-circular indeterminate :width="4" />
            </template>
          </v-btn>
        </v-card-actions>
      </v-card>
    </template>
  </v-dialog>
</template>

<script setup>
const loading = ref()
const feedbackInput = ref()

const form = ref('form')
const valid = ref(false)

const validationRules = ref([
  value => {
    if (value && value.length <= 1000) return true
    return 'Please provide valid feedback.'
  }
])

const BACKEND_URL = useRuntimeConfig().public.backendURL

const sendFeedback = async () => {
  loading.value = true
  form.value.validate()
  if (!valid.value) {
    console.warn('Validation of the form failed.')
    return
  }
  const formData = new FormData()
  formData.append("feedback", feedbackInput.value)
  await fetch(`http://${BACKEND_URL}/feedback`, {
    method: "POST",
    body: formData
  }).then(() => {
    loading.value = false
    feedbackInput.value = null
  })
}
</script>
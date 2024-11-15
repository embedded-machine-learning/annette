<template>
  <div>
    <h1>ONNX Datei hochladen</h1>
    <div v-if="message" class="message">
      {{ message }}
    </div>
    <form @submit.prevent="uploadFile">
      <input type="file" ref="fileInput" @change="onFileChange" />
      <button type="submit">Hochladen</button>
    </form>
  </div>
</template>

<script>
export default {
  data() {
    return {
      selectedFile: null,
      message: "",
    };
  },
  methods: {
    onFileChange(event) {
      this.selectedFile = event.target.files[0];
    },
    async uploadFile() {
      if (!this.selectedFile) {
        this.message = "Keine Datei ausgewählt";
        return;
      }

      const formData = new FormData();
      formData.append("file", this.selectedFile);

      try {
        const response = await fetch("http://127.0.0.1:5000/upload", {
          method: "POST",
          body: formData,
        });
        
        if (response.ok) {
          this.message = "Datei erfolgreich hochgeladen";
          this.selectedFile = null;
          this.$refs.fileInput.value = "";
        } else {
          this.message = "Ungültiges Dateiformat. Bitte laden Sie eine .onnx Datei hoch.";
        }
      } catch (error) {
        console.error("Fehler beim Hochladen der Datei:", error);
        this.message = "Es gab ein Problem beim Hochladen der Datei.";
      }
    },
  },
};
</script>

<style scoped>
.message {
  margin: 1em 0;
  color: #d9534f;
}
</style>

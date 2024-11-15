<template>
  <NuxtLayout>
    <v-app>
      <navigation-bar />
      <v-main>
        <v-container>
          <NuxtPage />
        </v-container>
      </v-main>
    </v-app>
  </NuxtLayout>
</template>

<script setup>
import NavigationBar from './components/navigation/navigation-bar.vue'

const databaseStore = useDatabaseStore()
await databaseStore.initialize()
</script>
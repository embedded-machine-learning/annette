<template>
    <v-navigation-drawer
        @update:rail="(railState) => changeLogoImage(railState)"
        expand-on-hover
        close-delay="50"
        rail
    >
        <v-img
            :src="logoImage"
        ></v-img>
        <v-divider></v-divider>

        <v-list nav>
            <v-list-item v-for="(item, i) in menuItems" :key="i" :to="item.route" rounded="lg">
                <template v-slot:prepend>
                    <v-icon :icon="item.icon"></v-icon>
                </template>
                <v-list-item-title v-text="item.title"></v-list-item-title>
            </v-list-item>
        </v-list>
        <template v-slot:append>
            <FeedbackDialog />
        </template>
    </v-navigation-drawer>
</template>

<script setup>
import { ref } from 'vue'
import AnnetteLogo from '../assets/annette-logo.png'
import AnnetteBanner from '../assets/annette-banner.png'
import FeedbackDialog from '../forms/feedback-dialog.vue'

var logoImage = AnnetteLogo
const menuItems = [
    {
        icon: 'mdi-rocket-launch',
        title: 'Estimation',
        route: '/'
    },
    {
        icon: 'mdi-compass',
        title: 'Exploration',
        route: '/exploration'
    },
    {
        icon: 'mdi-graph',
        title: 'Layer Analysis',
        route: '/layer-analysis'
    },
    {
        icon: 'mdi-database',
        title: 'Database',
        route: '/database'
    }
]

const changeLogoImage = (railState) => {
    logoImage = (railState) ? AnnetteLogo : AnnetteBanner
}
</script>
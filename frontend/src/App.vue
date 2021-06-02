<template>
  <div id="app">
    <!-- <img src="./assets/logo.png"> -->
    <Header></Header>
    <router-view/>
  </div>
</template>

<script>
import Header from './parts/Header.vue'
export default {
  name: 'app',
  components: {
    Header
  }
}
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#app {
  font-family: 'IBM Plex Sans', 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}
html,body { height:100%; }
body{
  margin: 0;
  padding: 0;
}
</style>
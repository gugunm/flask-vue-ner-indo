<template>
  <div class="header">
    <ul>
      <li><router-link to="/" exact>Home</router-link></li>
      <li><router-link to="/prediction">Prediction</router-link></li>
      <li><router-link to="/date-prediction">Date Prediction</router-link></li>
      <li><router-link to="/about">About</router-link></li>
    </ul>
  </div>
</template>


<style scoped>
ul {
  list-style-type: none;
  margin: 0;
  padding: 0 5%;
  overflow: hidden;
  border: 1px solid #e7e7e7;
  background-color: #f3f3f3;
}
li {
  float: left;
}
li a {
  display: block;
  color: #666;
  text-align: center;
  padding: 18px 25px;
  text-decoration: none;
}
li a:hover:not(.active) {
  background-color: #ddd;
}
.router-link-active {
  color: white;
  background-color: #1B2B47 !important;
}
</style>
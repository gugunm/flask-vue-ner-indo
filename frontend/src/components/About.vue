<template>
  <div class="page-about">
    <div class="foto">
      <img src="../assets/foto.jpg" />
    </div>
    <div class="biodata">
      <div class="text">
        <h3>Gugun Mediamer</h3>
        <p>19980101 202012 1 003</p>
        <hr>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page-about {
  padding: 20px 5%;
  display: flex;
}

.foto {
  flex-grow: 1;
  height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.foto img{
  height: 200px;
  width: 200px;
  border-radius: 50%;
}

.biodata {
  flex-grow: 6;
}

.text {
  margin-top: 70px;
  text-align: left;
}
</style>

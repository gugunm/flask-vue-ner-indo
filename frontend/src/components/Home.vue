
<template>
  <div class="page-home">
    <div class="about">
      <div class="mentor">
        <img src="../assets/gugun.jpg" />
        <h3>Gugun Mediamer</h3>
        <p>19980101 202012 1 003</p>
        <p class="status"><i>Peserta</i></p>
        <p class="bidang">Magang di <br> Pengelolaan Data dan Informasi <br> pada Pusinfowas BPKP </p>
      </div>
    </div>
    <div class="project">
      <div class="logo">
        <img src="../assets/bpkp.png" class="logo-bpkp">
        <img src="../assets/kpu.png" class="logo-kpu">
        <img src="../assets/gia.png" class="logo-gia">
      </div>
      <h1>Aktualisasi CPNS 2021</h1>
      <h3>Optimalisasi Aplikasi Media Monitoring Opini Publik : <i>Name Entity Recognition</i> <br> pada Judul Berita Online untuk Analisis Isu Strategis Pengawasan di BPKP</h3>
    </div>
    <div class="pembimbing">
      <div class="mentor">
        <img src="../assets/pak-jafar.jpg" />
        <h3>Ja`far Numeiri</h3>
        <p>19760822 199903 1 001</p>
        <p class="status"><i>Mentor</i></p>
        <p class="bidang">Pengelolaan Data dan Informasi <br> pada Pusinfowas BPKP </p>
      </div>
      <div class="mentor">
        <img src="../assets/pak-john.jpg" />
        <h3>Dr. John Elim</h3>
        <p>19610628 198203 1 001</p>
        <p class="status"><i>Coach</i></p>
        <p class="bidang">Pusat Pendidikan <br> dan Pelatihan Pengawasan BPKP</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page-home {
  padding: 20px 5%;
  display: flex;
}

.about {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  margin-right: 10px;
}

.about img, .pembimbing img{
  height: 130px;
  width: 130px;
  border-radius: 50%;
}

.about h3, .pembimbing h3 {
  margin: 0px;
  margin-top: 10px;
}

.about p, .pembimbing p {
  margin: 0px;
  margin-top: 5px;
}

p.bidang {
  margin-top: 10px;
  font-size: 14px;
}

p.status {
  padding-top: 10px;
  font-weight: bold;
}

.pembimbing {
  flex-grow: 1;
  
  margin-left: 10px;
  height: 700px;
  display: flex;
  flex-direction: column;
  /* align-items: center;
  justify-content: center; */
  
}

.mentor {
  /* max-width: 300px; */
  background-color: whitesmoke;
  border-radius: 10px;
  width: 100%;
  padding: 20px;
  box-sizing: border-box;
  margin-bottom: 10px;
}

.project {
  flex-grow: 4;
  /* border: 0.5px solid #1B2B47; */
  border: 0.8px solid #1B2B47;
  border-radius: 10px;
  height: 600px;
}

.logo {
  /* text-align: left; */
  padding: 40px 5%;
}

.logo-bpkp {
  height: 40px;
}

.logo-kpu {
  height: 50px;
}

.logo-gia {
  height: 40px;
}

img {
  margin: 0px 20px;
}
</style>
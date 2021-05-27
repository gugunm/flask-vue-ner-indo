<template>
<div class='page-prediction'>
  <p class='judul'>Predict NER of News Title / Content</p>
  <!-- <p class='judul'>{{ predict.data.Hasanuddin }}</p> -->

  <div class="container content-prediction">
    <div class="sisi-kiri">
      <p class="judul-input-prediciton">Tuliskan judul / konten berita</p>
      <textarea v-model="inputan" id="" name="input-prediction" rows="10" cols="50" class="input-prediction"></textarea>
      <!-- <textarea type="text" name="input-prediction" id="" class="input-prediction"> -->
      <input type="button" value="Predict Content" class="btn-pediction" @click="greet">
    </div>
    <div class="sisi-kanan">
      <p>Keyword Terkait</p>
      <div class="result-prediction">
        <p>{{ hasil }}</p>
        <div v-for="(value, name, index) in predict.data" v-bind:key="index">
          <div v-if="value != 'O'">
            {{ name }} - {{ value }}
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
</template>

<script>
// import axios from 'axios'
export default {
  data () {
    return {
      predict: {
        data: {
          ',': 'O',
          ':': 'O',
          Bukan: 'O',
          Desak: 'O',
          Halu: 'O',
          Hasanuddin: 'L-PERSON',
          Jangan: 'O',
          Jokowi: 'U-PERSON',
          Mendesak: 'O',
          Mudah: 'O',
          Mundur: 'O',
          Perkara: 'O',
          Presiden: 'O',
          TB: 'B-PERSON',
          TPUA: 'O'
        }
      },
      inputan: '',
      hasil: ''
    }
  },
  methods: {
    greet: function (event) {
      // `this` inside methods points to the Vue instance
      this.hasil = this.inputan
    }
  }
  // methods: {
  //   getPredict () {
  //     this.predict = this.getPredictionFromBackend()
  //   },
  //   getPredictionFromBackend () {
  //     const path = `http://127.0.0.1:5000/api/predict`
  //     axios.get(path)
  //     .then(response => {
  //       this.predict = JSON.parse(JSON.stringify(response.data))
  //       console.log(this.predict)
  //     })
  //     .catch(error => {
  //       console.log(error)
  //     })
  //   }
  // },
  // created () {
  //   this.getPredict()
  // }
}
</script>

<style lang='css' scoped>
.page-prediction {
  padding: 0 15%;
}

.judul {
  padding: 30px 40px;
  text-align: left;
  background-color: #f7f7f7;
  font-size: 20px;
  font-weight: 700;
}

.page-prediction .content-prediction {
  box-sizing: border-box;
  display: flex;
}

.sisi-kiri, .sisi-kanan {
  box-sizing: border-box;
  background-color: #f7f7f7;
  padding: 20px 40px;
  height: 500px;
}

.sisi-kiri {
  /* flex-grow: 1; */
  width: 60%;
  margin-right: 10px;
}

.sisi-kanan {
  width: 40%;
  margin-left: 10px;
}

.sisi-kanan p {
  margin-top: 0;
}

.judul-input-prediciton {
  font-size: 18px;
  margin-top: 0;
}

.input-prediction {
  box-sizing: border-box;
  font-family: inherit;
  font-size: 16px;
  padding: 20px;
  width: 100%;
  height: 350px;
  border: none;
  resize: none;
  outline: none;
}

.btn-pediction {
  float: right;
  margin-top: 15px;
  box-sizing: border-box;
  padding: 15px 30px;
  cursor: pointer;
  border: none;
  font-size: 14px;
  background-color: #7EFFA2;
}

.result-prediction {
  background-color: white;
  padding: 20px;
}
</style>

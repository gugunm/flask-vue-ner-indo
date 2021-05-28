<template>
<div class='page-prediction'>
  <p class='judul'>Predict NER of News Title / Content</p>
  <!-- <p class='judul'>{{ predict.data.Hasanuddin }}</p> -->

  <div class="container content-prediction">
    <div class="sisi-kiri">
      <p class="judul-input-prediciton">Tuliskan judul / konten berita</p>
      <textarea v-model="inputan" id="" name="input-prediction" rows="10" cols="50" class="input-prediction" @keyup="textControl"></textarea>
      <input type="button" value="Predict Content" class="btn-pediction" @click="getPredict">
    </div>
    <div class="sisi-kanan">
      <p class="judul-hasil">Keyword Terkait</p>
      <div class="result-prediction">
        <table class="table-hasil">
          <thead>
            <tr>
              <th class="name">Kata</th>
              <th class="value">Tag</th>
            </tr>
          </thead>
          <div class="loading" v-if="loading === true">
            <beat-loader :loading="loading" :color="color"></beat-loader>
          </div>
          <div v-else>
            <div v-if="predict.data">
              <div v-for="(value, name, index) in predict.data" v-bind:key="index" >
                <div v-if="value != 'O'">
                  <tr >
                    <div >
                      <td class="name" >
                        {{ name }}
                      </td>
                      <td class="value">
                        {{ value.split("-")[1] }}
                      </td>
                    </div>
                  </tr>
                </div>
              </div>
            </div>
          </div>
        </table>
      </div>
    </div>
  </div>
</div>
</template>

<script>
import axios from 'axios'
import PulseLoader from 'vue-spinner/src/PulseLoader.vue'
import BeatLoader from 'vue-spinner/src/BeatLoader.vue'

export default {
  components: {
    PulseLoader,
    BeatLoader
  },
  data () {
    return {
      predict: {},
      inputan: '',
      loading: false,
      color: '#1290E4'
    }
  },
  methods: {
    textControl () {
      if (this.inputan === '') {
        this.predict = {}
      }
    },
    getPredict () {
      this.loading = true
      this.predict = this.getPredictionFromBackend(this.inputan)
    },
    getPredictionFromBackend () {
      const path = `http://127.0.0.1:5000/api/predict`
      return axios.post(path, {
        text: this.inputan
      })
      .then(response => {
        this.predict = JSON.parse(JSON.stringify(response.data))
        this.loading = false
      })
      .catch(error => {
        console.log(error)
      })
    }
  }
}
</script>

<style lang='css' scoped>
.page-prediction {
  padding: 0px 5%;
}

.judul {
  margin-top: 0px;
  padding: 30px 40px;
  text-align: left;
  background-color: #D1E3EF;
  font-size: 20px;
  font-weight: bold;
}

.page-prediction .content-prediction {
  box-sizing: border-box;
  display: flex;
}

.sisi-kiri, .sisi-kanan {
  box-sizing: border-box;
  background-color: #D1E3EF;
  padding: 20px 40px;
  height: 450px;
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
  text-align: left;
}

.input-prediction {
  box-sizing: border-box;
  font-family: inherit;
  font-size: 16px;
  padding: 20px;
  width: 100%;
  height: 300px;
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
  background-color: #1B2B47;
  color: white;
}

.judul-hasil {
  text-align: left;
}

.loading {
  margin-top: 50px;
}

.result-prediction {
  background-color: white;
  padding:10px;
  height: 350px;
  overflow: scroll;
}

/* table, td, th {
  border: 1px solid black;
} */

.table-hasil {
  box-sizing: border-box;
  display: inline-block;
  width: 100%;
  border-collapse: collapse;
}

.table-hasil tr {
  box-sizing: border-box;
  margin: 0;
  width: 100%;
  display: inline-block;
  /* padding: 5px 2px; */
}

.table-hasil th.name, .table-hasil th.value {
  padding: 10px 2px;
  display: inline-block;
}

.table-hasil td.name, .table-hasil td.value {
  margin-bottom: 3px;
  box-sizing: border-box;
  padding: 3px 5px;
  display: inline-block;
  text-align: left;

}

.table-hasil td.name, .table-hasil th.name {
  width: 180px;
}

.table-hasil td.value, .table-hasil th.value {
  width: 150px;
}

.table-hasil td.name {
  box-sizing: border-box;
  border: 1px solid #D1E3EF;
}

.table-hasil td.value {
  background-color: #D1E3EF;
  border: 1px solid #D1E3EF;
}
</style>

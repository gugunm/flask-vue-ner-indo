<template>
<div class='page-prediction'>
  <p class='judul'>Predict NER of News Title / Content</p>

  <div class="container content-prediction">
    <div class="sisi-kiri">
      <p class="judul-input-prediciton">Tuliskan judul / konten berita</p>
      <textarea v-model="inputan" id="" name="input-prediction" rows="10" cols="50" class="input-prediction" @keyup="textControl"></textarea>
      <input type="button" value="Predict Content" class="btn-pediction" @click="getPredict">
    </div>
    <div class="sisi-kanan">
      <p class="judul-hasil">Keyword Terkait</p>
      <div class="result-prediction">
        <div class="loading" v-if="loading === true">
          <beat-loader :loading="loading" :color="color"></beat-loader>
        </div>
        <div v-else>
          <div v-if="predict.data">
            <table class="table-hasil">
              <tr>
                <!-- <th class="idx">No</th> -->
                <th class="name">Kata</th>
                <th class="value">Tag</th>
              </tr>
              <!-- <div v-for="(value, name, index) in predict.data" v-bind:key="index"> -->
              <div v-for="(d, index) in predict.data" v-bind:key="index">
                <!-- <div v-if="d.tag != 'O'"> -->
                  <tr>
                    <div>
                      <td class="name">
                        {{ d.word }}
                      </td>
                      <td class="value">
                        {{ d.tag != 'O' ? d.tag.split("-")[1] : '-' }}
                      </td>
                    </div>
                  </tr>
                <!-- </div> -->
              </div>
            </table>
          </div>
        </div>
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
      // counter: 0
    }
  },
  methods: {
    // addCounter() {
    //   this.counter += 1
    // },
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
        // this.predict = JSON.parse(JSON.stringify(response.data))
        this.predict = response.data
        this.loading = false
        console.log(this.predict)
      })
      .catch(error => {
        console.log(error)
      })
    }
  }
}
</script>

<style lang="css" scoped>
.page-prediction {
  padding: 20px 5%;
  /* background-color: #f3f3f3;
  height: 860px; */
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

.sisi-kiri,
.sisi-kanan {
  box-sizing: border-box;
  background-color: #D1E3EF;
  padding: 20px 40px;
  height: 450px;
}

.sisi-kiri {
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
  padding: 10px;
  height: 350px;
  overflow: auto;
}

.table-hasil {
  box-sizing: border-box;
  display: inline-block;
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
}

.table-hasil tr {
  box-sizing: border-box;
  margin: 0;
  width: 100%;
  display: inline-block;
}

.table-hasil th.idx,
.table-hasil th.name,
.table-hasil th.value {
  text-align: center;
  margin-bottom: 10px;
}

.table-hasil td,
.table-hasil th {
  margin-bottom: 3px;
  box-sizing: border-box;
  padding: 3px 5px;
  display: inline-block;
  text-align: left;
}

.table-hasil td.idx,
.table-hasil th.idx {
  width: 10%;
}

.table-hasil td.idx {
  text-align: center;
}

.table-hasil td.name,
.table-hasil th.name {
  width: 45%;
}

.table-hasil td.value,
.table-hasil th.value {
  width: 40%;
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

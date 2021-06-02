<template>
<div class='page-prediction'>
  <p class='judul'>Auto Prediction News By Date</p>

  <div class="container content-prediction">
    <div class="sisi-kiri">
      <p class="judul-input-prediciton">Pilih Tanggal</p>
      <date-picker 
        class="date-picker" 
        v-model="date" 
        value-type="format" 
        format="YYYY-MM-DD"
        @change="getNewsData">
      </date-picker>
    </div>
    <div class="sisi-tengah">
      <p class="judul-input-prediciton">Berita</p>
      <p>{{ date }}</p>
      <p v-if="newsByDate != null">{{ newsByDate.data.berita }}</p>
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
                <th class="name">Kata</th>
                <th class="value">Tag</th>
              </tr>
              <div v-for="(value, name, index) in predict.data" v-bind:key="index">
                <div v-if="value != 'O'">
                  <tr>
                    <div>
                      <td class="name">
                        {{ name }}
                      </td>
                      <td class="value">
                        {{ value.split("-")[1] }}
                      </td>
                    </div>
                  </tr>
                </div>
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
import DatePicker from 'vue2-datepicker'
import 'vue2-datepicker/index.css'

export default {
  components: {
    PulseLoader,
    BeatLoader,
    DatePicker
  },
  data () {
    return {
      predict: {},
      inputan: '',
      loading: false,
      color: '#1290E4',
      date: '2019-10-09',
      newsByDate: null
    }
  },
  methods: {
    getNewsData () {
      this.newsByDate = {
        data: {
          berita: 'KPK hari ini'
        }
      }
    },
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

<style lang="css" scoped>
.page-prediction {
  padding: 20px 5%;
  /* background-color: #f3f3f3;
  height: 860px; */
}

.judul {
  margin-top: 0px;
  padding: 30px 0px;
  text-align: left;
  /* background-color: #D1E3EF; */
  font-size: 20px;
  font-weight: bold;
}

.page-prediction .content-prediction {
  box-sizing: border-box;
  display: flex;
}

.sisi-tengah,
.sisi-kanan {
  box-sizing: border-box;
  background-color: #D1E3EF;
}

.sisi-kiri {
  box-sizing: border-box;
  width: 20%;
  margin-right: 10px;
  height: 400px;
  padding: 20px 40px;
}

.sisi-tengah {
  padding: 20px 40px;
  width: 40%;
  height: 600px;
}

.sisi-kanan {
  padding: 20px 40px;
  width: 35%;
  margin-left: 10px;
  height: 600px;
}

.sisi-kanan p {
  margin-top: 0;
}

.date-picker {
  display: block;
  width: 100%;
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
  /* background-color: white; */
  padding: 10px;
  height: 500px;
  overflow: scroll;
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

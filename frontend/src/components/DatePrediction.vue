<template>
<div class='page-prediction'>
  <p class='judul'>Auto Prediction News By Date</p>

  <div class="container content-prediction">
    <div class="sisi-kiri">
      <p>Pilih Tanggal</p>
      <date-picker 
        class="date-picker" 
        v-model="date" 
        value-type="format" 
        format="YYYY-MM-DD"
        :disabled-date="disabledAfterToday"
        >
      </date-picker>
      <p>Jumlah Berita</p>
      <div class="jml-berita">
        <input type="text" name="jml-berita" v-model="limit" @keyup.enter="getPredictByDate">
      </div>
      <div class="btn-kiri">
        <input 
          type="button" 
          class="btn-run" 
          value="Run"
          @click="getPredictByDate"
        >
        <input 
          type="button" 
          class="btn-reset" 
          value="Reset"
        >
      </div>
    </div>
    <div class="sisi-tengah">
      <p class="judul-tengah">Berita Terkait</p>
      <!-- <div class="card">
        <div class="card-media">
          <span class="icon-media">
            <i class="fas fa-user-edit fa-sm"></i>
          </span>
          <p>Kontan</p>
        </div>
        <p class="card-judul">KPK Hari ini Dimatikan</p>
        <div class="card-bottom">
          <div class="card-pubday">
            <span class="icon-clock">
              <i class="far fa-clock fa-sm"></i>
            </span>
            <p>2021-06-02</p>
          </div>
          <a class="card-link" href="https://google.com" target="_blank">
            <span>
              <i class="fas fa-link"></i>
            </span>
            <p>Source link</p>
          </a>
        </div>
      </div> -->
      <div v-if="predict.data" class="list-berita">
        <div v-for="(news, idx) in predict.data.news" :key="idx">
          <div 
            class="card"
            @click="showClikedNer(idx)"
          >
            <div class="card-media">
              <span class="icon-media">
                <i class="fas fa-user-edit fa-sm"></i>
              </span>
              <p>{{ news.media }}</p>
            </div>
            <p class="card-judul">{{ news.title }}</p>
            <div class="card-bottom">
              <div class="card-pubday">
                <span class="icon-clock">
                  <i class="far fa-clock fa-sm"></i>
                </span>
                <p>{{ news.pubday }}</p>
              </div>
              <a class="card-link" :href="news.url" target="_blank">
                <span>
                  <i class="fas fa-link"></i>
                </span>
                <p>Source link</p>
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="sisi-kanan">
      <p class="judul-hasil">Hasil Prediksi <i>Name Entity Recognation</i></p>
      <div class="result-prediction">
        <div class="loading" v-if="loading === true">
          <beat-loader :loading="loading" :color="color"></beat-loader>
        </div>
        <div v-else>
          <table v-if="predict.data" class="table-hasil">
            <tr>
              <th class="name">Kata</th>
              <th class="value">Tag</th>
            </tr>
            <div v-for="(d, index) in predict.data.news[idx_result].ner_title" v-bind:key="index">
              <div v-if="d.tag != 'O'">
                <tr>
                  <!-- <div> -->
                    <td class="name">
                      {{ d.word }}
                    </td>
                    <td class="value">
                      {{ d.tag != 'O' ? d.tag.split("-")[1] : '-' }}
                    </td>
                  <!-- </div> -->
                </tr>
              </div>
            </div>
          </table>
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
      idx_result: 0,
      inputan: '',
      loading: false,
      color: '#1290E4',
      date: new Date().toISOString().split('T')[0],
      newsByDate: null,
      limit: 100
    }
  },
  methods: {
    // limitControl () {
    //   if (this.inputan === '' || this.inputan === 0) {
    //     this.limit = 10
    //   }
    // },
    showClikedNer (idx) {
      this.idx_result = idx
    },
    disabledAfterToday (date) {
      const today = new Date()
      today.setHours(0, 0, 0, 0)

      return date > new Date(today.getTime())
    },
    getPredictByDate () {
      this.loading = true
      this.predict = this.getPredictionFromBackend(this.date)
    },
    getPredictionFromBackend () {
      const path = `http://127.0.0.1:5000/api/predict-by-date`
      return axios.get(path, {
        params: {
          tgl: this.date,
          limit: this.limit
        }
      })
      .then(response => {
        // this.predict = JSON.parse(JSON.stringify(response.data))
        this.predict = response.data
        this.loading = false
        console.log('DATAAAA')
        console.log(response.data)
      })
      .catch(error => {
        console.log(error)
      })
    }
  }
}
</script>

<style lang="css" scoped>
/* Global */
.page-prediction {
  padding: 20px 5%;
}

.judul {
  margin-top: 0px;
  padding: 30px 0px;
  text-align: left;
  font-size: 20px;
  font-weight: bold;
}

.page-prediction .content-prediction {
  box-sizing: border-box;
  display: flex;
}
/* ========= */

.sisi-tengah,
.sisi-kanan {
  box-sizing: border-box;
}

.sisi-kanan p, 
.sisi-tengah .judul-tengah {
  margin-top: 0;
  text-align: left;
}

/* SISI KIRI */
.sisi-kiri {
  box-sizing: border-box;
  width: 20%;
  margin-right: 20px;
  height: 400px;
}

.sisi-kiri p {
  text-align: left;
}

.date-picker {
  display: block;
  width: 100%;
}

.jml-berita input {
  box-sizing: border-box;
  padding: 5px;
  width: 100%; 
  color: #555;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-shadow: inset 0 1px 1px rgb(0 0 0 / 8%);
  height: 34px;
  padding: 6px 30px;
  padding-left: 10px;
}

.btn-kiri {
  margin-top: 20px;
}

.btn-run,
.btn-reset {
  width: 49%; 
  padding: 10px;
  cursor: pointer;
  margin-bottom: 5px;
}

.btn-run {
  background-color: #1B2B47;
  border-color: #1B2B47;
  color: white;
  border-radius: 3px;
}

.btn-run:hover {
  background-color: #284069;
  border-color: #284069;
  border-radius: 3px;
}

.btn-reset {
  background-color: white;
  border-color: #1B2B47;
  color: #1B2B47;
  border-radius: 3px;
}

.btn-reset:hover {
  background-color: #D1E3EF;
}
/* ============= */

/* SISI TENGAAH */
.sisi-tengah {
  padding: 20px 20px;
  width: 40%;
  height: 600px;
}

.list-berita {
  padding-right: 20px;
  height: 700px;
  overflow: auto;
}

.card {
  background-color: #f8f8f8;
  padding: 25px;
  margin-bottom: 5px;
  border-radius: 10px;
}

.card p {
  margin: 0;
}

.card-judul {
  font-size: 18px;
  text-align: left;
  font-weight: bold;
  cursor: pointer;
}

.card-media {
  text-align: left;
  margin-bottom: 10px;
}

.card-media .icon-media {
  padding-right: 2px;
}

.card-media p {
  display: inline-block;
}

.card-bottom {
  margin-top: 5px;
  margin-bottom: 20px;
}

.card-pubday {
  float: left;
  width: 50%;
  text-align: left;
}

.card-pubday .icon-clock {
  font-size: 15px;
  padding-right: 2px;
}

.card-pubday p {
  font-size: 12px;
  display: inline-block;
}

.card-link {
  float: right;
  width: 50%;
  text-align: right;
}

.card-link p {
  display: inline-block;
  font-size: 12px;
}

.card-link span {
  font-size: 10px;
}

a {
  color: #1B2B47;
}

/* ============= */


/* SISI KANAN - HASIL */
.sisi-kanan {
  padding: 20px;
  width: 35%;
  margin-left: 10px;
  height: 600px;
}

.judul-hasil {
  padding-left: 5px;
  text-align: left;
  margin-bottom: 10px;
}

.loading {
  margin-top: 50px;
}

.result-prediction {
  height: 500px;
  overflow: auto;
}

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
  display: block;
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
  width: 49%;
}

.table-hasil td.value,
.table-hasil th.value {
  width: 49%;
}

.table-hasil td.name {
  box-sizing: border-box;
  border: 1px solid #1B2B47;
}

.table-hasil td.value {
  background-color: #1B2B47;
  border: 1px solid #1B2B47;
  color: white;
}
/* ============= */
</style>

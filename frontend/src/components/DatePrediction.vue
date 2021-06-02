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
        :disabled-date="disabledAfterToday"
        >
      </date-picker>
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
      <p class="judul-input-prediciton">Berita</p>
      <div v-if="predict.data">
        <div v-for="(news, index) in predict.data.news" :key="index">
          <div 
            class="card"
            @click="showClikedNer(index)"
          >
            <p>{{ news.media }}</p>
            <p>{{ news.pubday }}</p>
            <p>{{ news.title }}</p>
          </div>
        </div>
      </div>
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
              <div v-for="(d, index) in predict.data.news[idx_result].ner_title" v-bind:key="index">
                <!-- <div v-if="value != 'O'"> -->
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
      newsByDate: null
    }
  },
  methods: {
    showClikedNer (idx) {
      this.index = idx
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
          tgl: this.date
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

.btn-kiri {
  margin-top: 10px;
}

.btn-run,
.btn-reset {
  width: 130px; 
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


/* SISI TENGAAH */
.card {
  background-color: white;
  cursor: pointer;
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

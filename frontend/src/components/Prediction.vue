<template>
  <div>
    <p>{{ predict.data.TB }}</p>
  </div>
</template>

<script>
import axios from 'axios'
export default {
  data () {
    return {
      predict: {}
    }
  },
  methods: {
    getPredict () {
      this.predict = this.getPredictionFromBackend()
    },
    getPredictionFromBackend () {
      const path = `http://127.0.0.1:5000/api/predict`
      axios.get(path)
      .then(response => {
        this.predict = JSON.parse(JSON.stringify(response.data))
        console.log(this.predict)
      })
      .catch(error => {
        console.log(error)
      })
    }
  },
  created () {
    this.getPredict()
  }
}
</script>
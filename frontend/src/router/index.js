import Vue from 'vue'
import Router from 'vue-router'
const routerOptions = [
  { path: '/', component: 'Home' },
  { path: '/prediction', component: 'Prediction' },
  { path: '/date-prediction', component: 'DatePrediction' },
  { path: '/about', component: 'About' },
  { path: '*', component: 'NotFound' }
]
const routes = routerOptions.map(route => {
  return {
    ...route,
    component: () => import(`@/components/${route.component}.vue`)
  }
})
Vue.use(Router)
export default new Router({
  routes,
  mode: 'history'
})

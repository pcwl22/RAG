import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import { initializeAuth } from './auth/oidc.js'

const bootstrap = async () => {
  await initializeAuth()
  createApp(App).mount('#app')
}

bootstrap().catch((error) => {
  console.error('OIDC initialization failed', error)
  document.querySelector('#app').textContent = '身份认证初始化失败，请刷新页面或联系管理员。'
})

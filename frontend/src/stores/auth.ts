import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { getMe, login, logout, register } from '@/api/auth'
import type { User, LoginPayload, RegisterPayload } from '@/api/auth'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const loading = ref(false)
  const error = ref('')

  const isAuthenticated = computed(() => user.value !== null)
  const isAdmin = computed(() => user.value?.role === 'admin')
  const isModerator = computed(() => user.value?.role === 'moderator' || user.value?.role === 'admin')

  async function fetchUser() {
    try {
      const res = await getMe()
      user.value = res.data.user
    } catch {
      user.value = null
    }
  }

  async function doLogin(payload: LoginPayload) {
    loading.value = true
    error.value = ''
    try {
      const res = await login(payload)
      user.value = res.data.user
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { error?: string } } })?.response?.data?.error
      error.value = msg || 'Login failed'
      throw err
    } finally {
      loading.value = false
    }
  }

  async function doRegister(payload: RegisterPayload) {
    loading.value = true
    error.value = ''
    try {
      const res = await register(payload)
      user.value = res.data.user
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { error?: string } } })?.response?.data?.error
      error.value = msg || 'Registration failed'
      throw err
    } finally {
      loading.value = false
    }
  }

  async function doLogout() {
    await logout()
    user.value = null
  }

  return { user, loading, error, isAuthenticated, isAdmin, isModerator, fetchUser, doLogin, doRegister, doLogout }
})

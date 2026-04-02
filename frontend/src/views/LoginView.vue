<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { useRouter, useRoute } from 'vue-router'

const auth = useAuthStore()
const router = useRouter()
const route = useRoute()

const email = ref('')
const password = ref('')
const oauthError = ref('')

onMounted(() => {
  const error = route.query.error as string
  if (error) {
    const errorMessages: Record<string, string> = {
      no_code: 'GitHub authorization was cancelled',
      exchange_failed: 'Failed to authenticate with GitHub',
      user_fetch_failed: 'Failed to get GitHub user info',
      create_failed: 'Failed to create account',
      token_failed: 'Failed to create session',
    }
    oauthError.value = errorMessages[error] || 'GitHub login failed'
  }
})

async function handleSubmit() {
  try {
    await auth.doLogin({ email: email.value, password: password.value })
    const redirect = (route.query.redirect as string) || '/problems'
    router.push(redirect)
  } catch {
    // error is set in the store
  }
}

function loginWithGitHub() {
  const redirect = (route.query.redirect as string) || '/problems'
  window.location.href = `/api/auth/github?redirect=${encodeURIComponent(redirect)}`
}
</script>

<template>
  <div class="auth-page">
    <div class="auth-card">
      <h2 class="auth-title">Login</h2>

      <div v-if="auth.error || oauthError" class="error-msg">{{ auth.error || oauthError }}</div>

      <button type="button" class="btn btn-github" @click="loginWithGitHub">
        <svg class="github-icon" viewBox="0 0 16 16" width="20" height="20">
          <path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
        </svg>
        Continue with GitHub
      </button>

      <div class="divider">
        <span>or</span>
      </div>

      <form @submit.prevent="handleSubmit" class="auth-form">
        <div class="form-group">
          <label for="email">Email</label>
          <input
            id="email"
            v-model="email"
            type="email"
            placeholder="you@example.com"
            required
          />
        </div>

        <div class="form-group">
          <label for="password">Password</label>
          <input
            id="password"
            v-model="password"
            type="password"
            placeholder="Your password"
            required
          />
        </div>

        <button type="submit" class="btn btn-primary btn-block" :disabled="auth.loading">
          {{ auth.loading ? 'Logging in...' : 'Login' }}
        </button>
      </form>

      <p class="auth-footer">
        Don't have an account? <router-link to="/register">Register</router-link>
      </p>
    </div>
  </div>
</template>

<style scoped>
.auth-page {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: calc(100vh - 56px);
  padding: 24px;
}

.auth-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 40px;
  width: 100%;
  max-width: 400px;
}

.auth-title {
  font-size: 1.5rem;
  font-weight: 700;
  text-align: center;
  margin-bottom: 24px;
}

.error-msg {
  background: rgba(248, 81, 73, 0.1);
  border: 1px solid var(--error);
  color: var(--error);
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 14px;
  margin-bottom: 16px;
}

.auth-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.form-group label {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-secondary);
}

.btn-block {
  width: 100%;
  justify-content: center;
  padding: 12px;
}

.btn-block:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.auth-footer {
  text-align: center;
  margin-top: 20px;
  font-size: 14px;
  color: var(--text-secondary);
}

.btn-github {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  width: 100%;
  padding: 12px;
  background: #24292e;
  color: #fff;
  border: 1px solid #444;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-github:hover {
  background: #2f363d;
}

.github-icon {
  flex-shrink: 0;
}

.divider {
  display: flex;
  align-items: center;
  margin: 20px 0;
  color: var(--text-muted);
  font-size: 13px;
}

.divider::before,
.divider::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

.divider span {
  padding: 0 12px;
}
</style>

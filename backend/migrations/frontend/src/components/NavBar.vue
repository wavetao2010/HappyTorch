<script setup lang="ts">
import { useAuthStore } from '@/stores/auth'
import { useRouter } from 'vue-router'

const auth = useAuthStore()
const router = useRouter()

async function handleLogout() {
  await auth.doLogout()
  router.push({ name: 'login' })
}
</script>

<template>
  <header class="header">
    <router-link to="/" class="logo">
      <span class="logo-icon">&#x1F525;</span>
      <span>HappyTorch</span>
    </router-link>

    <nav class="nav-links">
      <router-link v-if="auth.isAuthenticated" to="/problems" class="nav-link">Problems</router-link>
      <router-link v-if="auth.isAuthenticated" to="/submissions" class="nav-link">Submissions</router-link>
      <router-link to="/leaderboard" class="nav-link">Leaderboard</router-link>
      <router-link v-if="auth.isAuthenticated" to="/problems/submit" class="nav-link">Submit Problem</router-link>
      <router-link v-if="auth.isModerator" to="/admin/review" class="nav-link nav-admin">Review Queue</router-link>
      <router-link v-if="auth.isAdmin" to="/admin/users" class="nav-link nav-admin">Users</router-link>
      <router-link v-if="auth.isAuthenticated" to="/profile" class="nav-link">Profile</router-link>
    </nav>

    <div class="header-actions">
      <template v-if="auth.isAuthenticated">
        <span v-if="auth.user?.points" class="user-points">{{ auth.user.points }} pts</span>
        <span class="user-info">{{ auth.user?.username }}</span>
        <button class="btn btn-secondary" @click="handleLogout">Logout</button>
      </template>
      <template v-else>
        <router-link to="/login" class="btn btn-secondary">Login</router-link>
        <router-link to="/register" class="btn btn-primary">Register</router-link>
      </template>
    </div>
  </header>
</template>

<style scoped>
.header {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text-primary);
}

.logo:hover {
  color: var(--text-primary);
}

.logo-icon {
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, #ff6b35 0%, #f7931a 100%);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
}

.nav-links {
  display: flex;
  gap: 16px;
}

.nav-link {
  color: var(--text-secondary);
  font-size: 14px;
  font-weight: 500;
  padding: 6px 12px;
  border-radius: 6px;
  transition: all 0.2s;
}

.nav-link:hover,
.nav-link.router-link-active {
  color: var(--text-primary);
  background: var(--bg-tertiary);
}

.nav-admin {
  color: var(--warning);
}

.nav-admin:hover,
.nav-admin.router-link-active {
  color: var(--warning);
  background: rgba(210, 153, 34, 0.1);
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.user-points {
  color: var(--accent);
  font-size: 13px;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
}

.user-info {
  color: var(--text-secondary);
  font-size: 14px;
  font-weight: 500;
}
</style>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { getUserProfile } from '@/api/community'
import type { PublicUserProfile } from '@/api/community'

const route = useRoute()
const profile = ref<PublicUserProfile | null>(null)
const loading = ref(true)
const error = ref('')

async function fetchProfile() {
  loading.value = true
  error.value = ''
  try {
    const username = route.params.username as string
    const res = await getUserProfile(username)
    profile.value = res.data.user
  } catch {
    error.value = 'User not found.'
  } finally {
    loading.value = false
  }
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'long',
    year: 'numeric',
  })
}

onMounted(fetchProfile)
watch(() => route.params.username, fetchProfile)
</script>

<template>
  <div class="user-profile-page">
    <div v-if="loading" class="loading">Loading profile...</div>
    <div v-else-if="error" class="error">{{ error }}</div>

    <template v-else-if="profile">
      <div class="profile-card">
        <div class="profile-avatar">{{ (profile.display_name || profile.username).charAt(0).toUpperCase() }}</div>
        <div class="profile-info">
          <h1>{{ profile.display_name || profile.username }}</h1>
          <div class="profile-username">@{{ profile.username }}</div>
          <div v-if="profile.bio" class="profile-bio">{{ profile.bio }}</div>
          <div class="profile-meta">
            <span v-if="profile.role !== 'user'" class="role-badge">{{ profile.role }}</span>
            <span class="join-date">Joined {{ formatDate(profile.created_at) }}</span>
          </div>
        </div>
      </div>

      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-value">{{ profile.points }}</div>
          <div class="stat-label">Points</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ profile.problems_solved }}</div>
          <div class="stat-label">Problems Solved</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ profile.contributions }}</div>
          <div class="stat-label">Contributions</div>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.user-profile-page {
  max-width: 700px;
  margin: 0 auto;
  padding: 32px 24px;
}

.loading,
.error {
  text-align: center;
  color: var(--text-secondary);
  padding: 40px;
  font-size: 14px;
}

.profile-card {
  display: flex;
  align-items: flex-start;
  gap: 24px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 32px;
  margin-bottom: 24px;
}

.profile-avatar {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: var(--accent);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 32px;
  font-weight: 700;
  flex-shrink: 0;
}

.profile-info h1 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 4px;
}

.profile-username {
  color: var(--text-secondary);
  font-size: 14px;
  margin-bottom: 8px;
}

.profile-bio {
  color: var(--text-primary);
  font-size: 14px;
  line-height: 1.5;
  margin-bottom: 12px;
}

.profile-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 13px;
  color: var(--text-secondary);
}

.role-badge {
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  background: rgba(88, 166, 255, 0.15);
  color: var(--accent);
  text-transform: capitalize;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.stat-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  text-align: center;
}

.stat-card .stat-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--accent);
  margin-bottom: 4px;
}

.stat-card .stat-label {
  font-size: 12px;
  color: var(--text-secondary);
}

@media (max-width: 640px) {
  .profile-card {
    flex-direction: column;
    align-items: center;
    text-align: center;
  }

  .profile-meta {
    justify-content: center;
  }
}
</style>

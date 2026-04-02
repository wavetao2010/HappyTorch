<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { getProgress, getSubmissions } from '@/api/problems'
import type { UserProgress, Submission } from '@/api/problems'

const auth = useAuthStore()
const progress = ref<UserProgress | null>(null)
const recentSubmissions = ref<Submission[]>([])
const loading = ref(true)

onMounted(async () => {
  try {
    const [progressRes, subsRes] = await Promise.all([getProgress(), getSubmissions(1, 5)])
    progress.value = progressRes.data.progress
    recentSubmissions.value = subsRes.data.submissions ?? []
  } catch {
    // Progress/submissions not available
  } finally {
    loading.value = false
  }
})

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}
</script>

<template>
  <div class="profile-page">
    <h1>Profile</h1>

    <div v-if="loading" class="loading">Loading profile...</div>

    <template v-else>
      <!-- User info -->
      <div class="user-card">
        <div class="user-avatar">{{ auth.user?.username?.charAt(0)?.toUpperCase() ?? '?' }}</div>
        <div class="user-details">
          <h2>{{ auth.user?.username }}</h2>
          <div v-if="progress" class="user-stats">
            <div class="stat">
              <span class="stat-value">{{ progress.points }}</span>
              <span class="stat-label">Points</span>
            </div>
            <div class="stat">
              <span class="stat-value">{{ progress.solved }}</span>
              <span class="stat-label">Solved</span>
            </div>
            <div class="stat">
              <span class="stat-value">#{{ progress.rank }}</span>
              <span class="stat-label">Rank</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Progress by difficulty -->
      <div v-if="progress" class="progress-section">
        <h3>Progress</h3>
        <div class="difficulty-progress">
          <div class="progress-row">
            <span class="diff-label easy">Easy</span>
            <div class="progress-bar-bg">
              <div
                class="progress-bar easy"
                :style="{ width: progress.by_difficulty.easy.total > 0 ? (progress.by_difficulty.easy.solved / progress.by_difficulty.easy.total * 100) + '%' : '0%' }"
              />
            </div>
            <span class="progress-count">{{ progress.by_difficulty.easy.solved }}/{{ progress.by_difficulty.easy.total }}</span>
          </div>
          <div class="progress-row">
            <span class="diff-label medium">Medium</span>
            <div class="progress-bar-bg">
              <div
                class="progress-bar medium"
                :style="{ width: progress.by_difficulty.medium.total > 0 ? (progress.by_difficulty.medium.solved / progress.by_difficulty.medium.total * 100) + '%' : '0%' }"
              />
            </div>
            <span class="progress-count">{{ progress.by_difficulty.medium.solved }}/{{ progress.by_difficulty.medium.total }}</span>
          </div>
          <div class="progress-row">
            <span class="diff-label hard">Hard</span>
            <div class="progress-bar-bg">
              <div
                class="progress-bar hard"
                :style="{ width: progress.by_difficulty.hard.total > 0 ? (progress.by_difficulty.hard.solved / progress.by_difficulty.hard.total * 100) + '%' : '0%' }"
              />
            </div>
            <span class="progress-count">{{ progress.by_difficulty.hard.solved }}/{{ progress.by_difficulty.hard.total }}</span>
          </div>
        </div>
      </div>

      <!-- Recent submissions -->
      <div class="recent-section">
        <div class="section-header">
          <h3>Recent Submissions</h3>
          <router-link to="/submissions" class="view-all">View All</router-link>
        </div>

        <div v-if="recentSubmissions.length === 0" class="empty">No submissions yet.</div>

        <div v-else class="recent-list">
          <div v-for="s in recentSubmissions" :key="s.id" class="recent-item">
            <router-link :to="`/problems/${s.problem_slug}`" class="recent-title">
              {{ s.problem_title }}
            </router-link>
            <div class="recent-meta">
              <span :class="['status-badge', s.success ? 'success' : 'failed']">
                {{ s.success ? 'Accepted' : 'Failed' }}
              </span>
              <span class="recent-date">{{ formatDate(s.created_at) }}</span>
            </div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.profile-page {
  max-width: 700px;
  margin: 0 auto;
  padding: 32px 24px;
}

.profile-page h1 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 24px;
}

.loading,
.empty {
  text-align: center;
  color: var(--text-secondary);
  padding: 24px;
  font-size: 14px;
}

.user-card {
  display: flex;
  align-items: center;
  gap: 20px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 24px;
}

.user-avatar {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: var(--accent);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  font-weight: 700;
  flex-shrink: 0;
}

.user-details h2 {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 12px;
}

.user-stats {
  display: flex;
  gap: 24px;
}

.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stat-value {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--accent);
}

.stat-label {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 2px;
}

.progress-section {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 24px;
}

.progress-section h3,
.section-header h3 {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 16px;
}

.difficulty-progress {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.progress-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.diff-label {
  font-size: 13px;
  font-weight: 500;
  min-width: 60px;
  text-transform: capitalize;
}

.diff-label.easy { color: var(--easy); }
.diff-label.medium { color: var(--medium); }
.diff-label.hard { color: var(--hard); }

.progress-bar-bg {
  flex: 1;
  height: 8px;
  background: var(--bg-tertiary);
  border-radius: 4px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s;
}

.progress-bar.easy { background: var(--easy); }
.progress-bar.medium { background: var(--medium); }
.progress-bar.hard { background: var(--hard); }

.progress-count {
  font-size: 13px;
  color: var(--text-secondary);
  font-family: 'JetBrains Mono', monospace;
  min-width: 40px;
  text-align: right;
}

.recent-section {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.view-all {
  font-size: 13px;
  color: var(--accent);
}

.recent-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.recent-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
}

.recent-item:last-child {
  border-bottom: none;
}

.recent-title {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.recent-title:hover {
  color: var(--accent);
}

.recent-meta {
  display: flex;
  align-items: center;
  gap: 12px;
}

.status-badge {
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
}

.status-badge.success {
  background: rgba(63, 185, 80, 0.15);
  color: var(--success);
}

.status-badge.failed {
  background: rgba(248, 81, 73, 0.15);
  color: var(--error);
}

.recent-date {
  font-size: 12px;
  color: var(--text-muted);
}
</style>

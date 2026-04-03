<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { getLeaderboard } from '@/api/community'
import type { LeaderboardEntry } from '@/api/community'

const auth = useAuthStore()
const entries = ref<LeaderboardEntry[]>([])
const loading = ref(true)
const activeTab = ref<'points' | 'problems_solved' | 'contributions'>('points')

async function fetchLeaderboard() {
  loading.value = true
  try {
    const res = await getLeaderboard(activeTab.value)
    entries.value = res.data.entries ?? []
  } catch {
    // Failed to load
  } finally {
    loading.value = false
  }
}

function switchTab(tab: 'points' | 'problems_solved' | 'contributions') {
  activeTab.value = tab
  fetchLeaderboard()
}

function rankClass(rank: number): string {
  if (rank === 1) return 'gold'
  if (rank === 2) return 'silver'
  if (rank === 3) return 'bronze'
  return ''
}

function scoreValue(entry: LeaderboardEntry): number {
  if (activeTab.value === 'problems_solved') return entry.problems_solved
  if (activeTab.value === 'contributions') return entry.contributions
  return entry.points
}

onMounted(fetchLeaderboard)
</script>

<template>
  <div class="leaderboard-page">
    <h1>Leaderboard</h1>

    <div class="tabs">
      <button
        :class="['tab', { active: activeTab === 'points' }]"
        @click="switchTab('points')"
      >Points</button>
      <button
        :class="['tab', { active: activeTab === 'problems_solved' }]"
        @click="switchTab('problems_solved')"
      >Problems Solved</button>
      <button
        :class="['tab', { active: activeTab === 'contributions' }]"
        @click="switchTab('contributions')"
      >Contributions</button>
    </div>

    <div v-if="loading" class="loading">Loading leaderboard...</div>

    <div v-else-if="entries.length === 0" class="empty">No entries yet.</div>

    <table v-else class="lb-table">
      <thead>
        <tr>
          <th class="rank-col">Rank</th>
          <th>User</th>
          <th class="score-col">Score</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="entry in entries"
          :key="entry.user_id"
          :class="{ 'current-user': entry.username === auth.user?.username }"
        >
          <td class="rank-col">
            <span :class="['rank', rankClass(entry.rank)]">{{ entry.rank }}</span>
          </td>
          <td>
            <router-link :to="`/users/${entry.username}`" class="user-cell">
              <div class="user-avatar-sm">{{ entry.username.charAt(0).toUpperCase() }}</div>
              <span class="user-name">{{ entry.username }}</span>
            </router-link>
          </td>
          <td class="score-col">{{ scoreValue(entry) }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.leaderboard-page {
  max-width: 700px;
  margin: 0 auto;
  padding: 32px 24px;
}

.leaderboard-page h1 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 20px;
}

.tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 24px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 4px;
}

.tab {
  flex: 1;
  padding: 8px 16px;
  border: none;
  background: transparent;
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 500;
  font-family: inherit;
  cursor: pointer;
  border-radius: 6px;
  transition: all 0.2s;
}

.tab:hover {
  color: var(--text-primary);
}

.tab.active {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.loading,
.empty {
  text-align: center;
  color: var(--text-secondary);
  padding: 40px;
  font-size: 14px;
}

.lb-table {
  width: 100%;
  border-collapse: collapse;
}

.lb-table th {
  text-align: left;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
}

.lb-table td {
  padding: 12px;
  border-bottom: 1px solid var(--border);
  font-size: 14px;
}

.lb-table tr:last-child td {
  border-bottom: none;
}

.rank-col {
  width: 60px;
  text-align: center;
}

.score-col {
  width: 100px;
  text-align: right;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
}

.rank {
  font-weight: 700;
  font-size: 15px;
}

.rank.gold { color: #ffd700; }
.rank.silver { color: #c0c0c0; }
.rank.bronze { color: #cd7f32; }

.current-user {
  background: rgba(88, 166, 255, 0.08);
}

.user-cell {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--text-primary);
}

.user-cell:hover {
  color: var(--accent);
}

.user-avatar-sm {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: var(--accent);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 700;
  flex-shrink: 0;
}

.user-name {
  font-weight: 500;
}
</style>

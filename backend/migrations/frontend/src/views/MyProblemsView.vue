<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { getMyProblems } from '@/api/community'
import type { CommunityProblem } from '@/api/community'

const router = useRouter()
const problems = ref<CommunityProblem[]>([])
const loading = ref(true)

onMounted(async () => {
  try {
    const res = await getMyProblems()
    problems.value = res.data.problems ?? []
  } catch {
    // Failed to load
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

function editProblem(p: CommunityProblem) {
  router.push({ path: '/problems/submit', query: { edit: p.id } })
}
</script>

<template>
  <div class="my-problems-page">
    <div class="page-header">
      <h1>My Problems</h1>
      <router-link to="/problems/submit" class="btn btn-primary">+ Submit New</router-link>
    </div>

    <div v-if="loading" class="loading">Loading your problems...</div>

    <div v-else-if="problems.length === 0" class="empty">
      You haven't submitted any problems yet.
      <router-link to="/problems/submit">Submit your first problem</router-link>
    </div>

    <div v-else class="problems-list">
      <div v-for="p in problems" :key="p.id" class="problem-row">
        <div class="problem-info">
          <div class="problem-title">{{ p.title }}</div>
          <div class="problem-meta">
            <span :class="['diff-badge', p.difficulty]">{{ p.difficulty }}</span>
            <span class="category">{{ p.category }}</span>
            <span class="date">{{ formatDate(p.created_at) }}</span>
          </div>
        </div>
        <div class="problem-actions">
          <span :class="['status-badge', p.status]">{{ p.status }}</span>
          <button
            v-if="p.status === 'pending' || p.status === 'rejected'"
            class="btn btn-secondary btn-sm"
            @click="editProblem(p)"
          >Edit</button>
        </div>
        <div v-if="p.review_note && p.status === 'rejected'" class="review-note">
          <strong>Review note:</strong> {{ p.review_note }}
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.my-problems-page {
  max-width: 800px;
  margin: 0 auto;
  padding: 32px 24px;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
}

.page-header h1 {
  font-size: 1.5rem;
  font-weight: 700;
}

.loading,
.empty {
  text-align: center;
  color: var(--text-secondary);
  padding: 40px;
  font-size: 14px;
}

.problems-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.problem-row {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}

.problem-title {
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 6px;
}

.problem-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  color: var(--text-secondary);
}

.diff-badge {
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  text-transform: capitalize;
}

.diff-badge.easy { background: rgba(63, 185, 80, 0.15); color: var(--easy); }
.diff-badge.medium { background: rgba(210, 153, 34, 0.15); color: var(--medium); }
.diff-badge.hard { background: rgba(248, 81, 73, 0.15); color: var(--hard); }

.problem-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.status-badge {
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  text-transform: capitalize;
}

.status-badge.pending { background: rgba(210, 153, 34, 0.15); color: var(--warning); }
.status-badge.approved { background: rgba(63, 185, 80, 0.15); color: var(--success); }
.status-badge.rejected { background: rgba(248, 81, 73, 0.15); color: var(--error); }

.btn-sm {
  padding: 4px 10px;
  font-size: 12px;
}

.review-note {
  width: 100%;
  padding: 10px 12px;
  background: rgba(248, 81, 73, 0.08);
  border-radius: 6px;
  font-size: 13px;
  color: var(--text-secondary);
}
</style>

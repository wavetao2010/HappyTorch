<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { getSubmissions } from '@/api/problems'
import type { Submission } from '@/api/problems'
import MonacoEditor from '@/components/MonacoEditor.vue'

const submissions = ref<Submission[]>([])
const total = ref(0)
const page = ref(1)
const limit = 20
const loading = ref(false)
const selectedSubmission = ref<Submission | null>(null)

const totalPages = computed(() => Math.max(1, Math.ceil(total.value / limit)))

async function fetchPage(p: number) {
  loading.value = true
  try {
    const res = await getSubmissions(p, limit)
    submissions.value = res.data.submissions ?? []
    total.value = res.data.total ?? 0
    page.value = p
  } finally {
    loading.value = false
  }
}

function formatDate(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function formatTime(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
}

onMounted(() => fetchPage(1))
</script>

<template>
  <div class="submissions-page">
    <h1>Submission History</h1>

    <div v-if="loading" class="loading">Loading submissions...</div>

    <div v-else-if="submissions.length === 0" class="empty">
      No submissions yet. Start solving problems!
    </div>

    <template v-else>
      <table class="submissions-table">
        <thead>
          <tr>
            <th>Problem</th>
            <th>Result</th>
            <th>Tests</th>
            <th>Time</th>
            <th>Date</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="s in submissions"
            :key="s.id"
            class="submission-row"
            @click="selectedSubmission = selectedSubmission?.id === s.id ? null : s"
          >
            <td class="problem-name">
              <router-link :to="`/problems/${s.problem_slug}`">
                {{ s.problem_title }}
              </router-link>
            </td>
            <td>
              <span :class="['status-badge', s.success ? 'success' : 'failed']">
                {{ s.success ? 'Accepted' : 'Failed' }}
              </span>
            </td>
            <td class="tests-col">{{ s.passed }}/{{ s.total }}</td>
            <td class="time-col">{{ s.execution_time }}ms</td>
            <td class="date-col">
              <span class="date-text">{{ formatDate(s.created_at) }}</span>
              <span class="time-text">{{ formatTime(s.created_at) }}</span>
            </td>
          </tr>
        </tbody>
      </table>

      <!-- Code viewer -->
      <div v-if="selectedSubmission" class="code-viewer">
        <div class="code-viewer-header">
          <h3>{{ selectedSubmission.problem_title }} - Submission Code</h3>
          <button class="btn btn-secondary" @click="selectedSubmission = null">Close</button>
        </div>
        <MonacoEditor :model-value="selectedSubmission.code" :read-only="true" height="300px" />
      </div>

      <!-- Pagination -->
      <div v-if="totalPages > 1" class="pagination">
        <button
          class="btn btn-secondary"
          :disabled="page <= 1"
          @click="fetchPage(page - 1)"
        >
          Previous
        </button>
        <span class="page-info">Page {{ page }} of {{ totalPages }}</span>
        <button
          class="btn btn-secondary"
          :disabled="page >= totalPages"
          @click="fetchPage(page + 1)"
        >
          Next
        </button>
      </div>
    </template>
  </div>
</template>

<style scoped>
.submissions-page {
  max-width: 900px;
  margin: 0 auto;
  padding: 32px 24px;
}

.submissions-page h1 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 24px;
}

.loading,
.empty {
  text-align: center;
  color: var(--text-secondary);
  padding: 40px;
  font-size: 14px;
}

.submissions-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.submissions-table th {
  text-align: left;
  padding: 10px 12px;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border-bottom: 1px solid var(--border);
}

.submission-row {
  cursor: pointer;
  transition: background 0.2s;
}

.submission-row:hover {
  background: var(--bg-secondary);
}

.submission-row td {
  padding: 12px;
  border-bottom: 1px solid var(--border);
}

.problem-name a {
  color: var(--text-primary);
  font-weight: 500;
}

.problem-name a:hover {
  color: var(--accent);
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

.tests-col,
.time-col {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: var(--text-secondary);
}

.date-col {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.date-text {
  color: var(--text-primary);
  font-size: 13px;
}

.time-text {
  color: var(--text-muted);
  font-size: 12px;
}

.code-viewer {
  margin-top: 20px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}

.code-viewer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.code-viewer-header h3 {
  font-size: 14px;
  font-weight: 600;
}

.pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  margin-top: 24px;
}

.page-info {
  font-size: 14px;
  color: var(--text-secondary);
}
</style>

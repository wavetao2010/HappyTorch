<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getPendingProblems, reviewProblem } from '@/api/admin'
import type { CommunityProblem } from '@/api/community'
import MonacoEditor from '@/components/MonacoEditor.vue'

const problems = ref<CommunityProblem[]>([])
const loading = ref(true)
const expandedId = ref<string | null>(null)
const reviewNote = ref('')
const reviewing = ref(false)

onMounted(async () => {
  try {
    const res = await getPendingProblems()
    problems.value = res.data.problems ?? []
  } catch {
    // Failed to load
  } finally {
    loading.value = false
  }
})

function toggleExpand(id: string) {
  expandedId.value = expandedId.value === id ? null : id
  reviewNote.value = ''
}

async function handleReview(id: string, action: 'approve' | 'reject') {
  reviewing.value = true
  try {
    await reviewProblem(id, action, reviewNote.value || undefined)
    problems.value = problems.value.filter((p) => p.id !== id)
    expandedId.value = null
    reviewNote.value = ''
  } catch {
    // Failed
  } finally {
    reviewing.value = false
  }
}
</script>

<template>
  <div class="review-page">
    <h1>Review Queue</h1>

    <div v-if="loading" class="loading">Loading pending problems...</div>

    <div v-else-if="problems.length === 0" class="empty">No problems pending review.</div>

    <div v-else class="review-list">
      <div v-for="p in problems" :key="p.id" class="review-item">
        <div class="review-header" @click="toggleExpand(p.id)">
          <div class="review-info">
            <span class="review-title">{{ p.title }}</span>
            <div class="review-meta">
              <span :class="['diff-badge', p.difficulty]">{{ p.difficulty }}</span>
              <span>{{ p.category }}</span>
            </div>
          </div>
          <span class="expand-icon">{{ expandedId === p.id ? '−' : '+' }}</span>
        </div>

        <div v-if="expandedId === p.id" class="review-detail">
          <div class="detail-section">
            <h4>Description</h4>
            <div class="detail-content">{{ p.description }}</div>
          </div>

          <div class="detail-section">
            <h4>Template Code</h4>
            <MonacoEditor :model-value="p.template_code" language="python" height="150px" :read-only="true" />
          </div>

          <div v-if="p.tests?.length" class="detail-section">
            <h4>Tests ({{ p.tests.length }})</h4>
            <div v-for="(t, i) in p.tests" :key="i" class="test-preview">
              <span class="test-label">{{ t.name }}</span>
              <MonacoEditor :model-value="t.code" language="python" height="100px" :read-only="true" />
            </div>
          </div>

          <div v-if="p.hint" class="detail-section">
            <h4>Hint</h4>
            <div class="detail-content">{{ p.hint }}</div>
          </div>

          <div class="review-actions">
            <input v-model="reviewNote" type="text" placeholder="Optional review note..." class="review-note-input" />
            <div class="action-buttons">
              <button
                class="btn btn-approve"
                :disabled="reviewing"
                @click="handleReview(p.id, 'approve')"
              >Approve</button>
              <button
                class="btn btn-reject"
                :disabled="reviewing"
                @click="handleReview(p.id, 'reject')"
              >Reject</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.review-page {
  max-width: 800px;
  margin: 0 auto;
  padding: 32px 24px;
}

.review-page h1 {
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

.review-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.review-item {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}

.review-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  cursor: pointer;
  transition: background 0.2s;
}

.review-header:hover {
  background: var(--bg-tertiary);
}

.review-title {
  font-size: 15px;
  font-weight: 600;
}

.review-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 4px;
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

.expand-icon {
  font-size: 20px;
  color: var(--text-secondary);
  font-weight: 300;
}

.review-detail {
  padding: 0 16px 16px;
  border-top: 1px solid var(--border);
}

.detail-section {
  margin-top: 16px;
}

.detail-section h4 {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

.detail-content {
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-primary);
  white-space: pre-wrap;
}

.test-preview {
  margin-top: 8px;
}

.test-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-secondary);
  margin-bottom: 4px;
  display: block;
}

.review-actions {
  margin-top: 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.review-note-input {
  flex: 1;
}

.action-buttons {
  display: flex;
  gap: 10px;
}

.btn-approve {
  background: var(--success);
  color: #fff;
  border: none;
  padding: 8px 20px;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  font-family: inherit;
}

.btn-reject {
  background: var(--error);
  color: #fff;
  border: none;
  padding: 8px 20px;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  font-family: inherit;
}

.btn-approve:hover { opacity: 0.9; }
.btn-reject:hover { opacity: 0.9; }
</style>

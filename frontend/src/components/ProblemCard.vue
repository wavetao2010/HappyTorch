<script setup lang="ts">
import type { Problem } from '@/api/problems'

defineProps<{
  problem: Problem
  status?: string
}>()

function difficultyClass(d: string) {
  return `difficulty-${d.toLowerCase()}`
}
</script>

<template>
  <router-link :to="`/problems/${problem.slug}`" class="problem-card">
    <div class="problem-info">
      <div class="title-row">
        <span v-if="status === 'solved'" class="status-icon solved" title="Solved">&#x2713;</span>
        <span v-else-if="status === 'attempted'" class="status-icon attempted" title="Attempted">&#x25CF;</span>
        <span v-else class="status-icon todo">&#x25CB;</span>
        <span class="problem-title">{{ problem.title }}</span>
      </div>
      <span class="problem-category">{{ problem.category }}</span>
    </div>
    <div class="problem-meta">
      <span :class="['difficulty-badge', difficultyClass(problem.difficulty)]">
        {{ problem.difficulty }}
      </span>
    </div>
  </router-link>
</template>

<style scoped>
.problem-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  transition: all 0.2s;
  color: var(--text-primary);
}

.problem-card:hover {
  background: var(--bg-hover);
  border-color: var(--accent);
  color: var(--text-primary);
}

.problem-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.title-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-icon {
  font-size: 14px;
  flex-shrink: 0;
}

.status-icon.solved {
  color: var(--success);
  font-weight: 700;
}

.status-icon.attempted {
  color: var(--warning);
}

.status-icon.todo {
  color: var(--text-muted);
}

.problem-title {
  font-weight: 500;
  font-size: 14px;
}

.problem-category {
  font-size: 12px;
  color: var(--text-muted);
  margin-left: 22px;
}

.problem-meta {
  display: flex;
  align-items: center;
  gap: 8px;
}

.difficulty-badge {
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  text-transform: capitalize;
}

.difficulty-easy {
  background: rgba(63, 185, 80, 0.15);
  color: var(--easy);
}

.difficulty-medium {
  background: rgba(210, 153, 34, 0.15);
  color: var(--medium);
}

.difficulty-hard {
  background: rgba(248, 81, 73, 0.15);
  color: var(--hard);
}
</style>

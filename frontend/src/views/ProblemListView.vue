<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useProblemsStore } from '@/stores/problems'
import { useAuthStore } from '@/stores/auth'
import { getProgress } from '@/api/problems'
import ProblemCard from '@/components/ProblemCard.vue'

const store = useProblemsStore()
const auth = useAuthStore()

const search = ref('')
const category = ref('')
const difficulty = ref('')
const solvedSlugs = ref<Set<string>>(new Set())
const attemptedSlugs = ref<Set<string>>(new Set())

const categories = ['基础层', '现代激活函数', '注意力机制', 'ML 基础与解码策略', 'LLM 推理组件', '参数高效微调', '条件调制 — Diffusion', '扩散模型训练', '完整架构', 'RLHF']
const difficulties = ['Easy', 'Medium', 'Hard']

function fetchFiltered() {
  store.fetchProblems({
    search: search.value || undefined,
    category: category.value || undefined,
    difficulty: difficulty.value || undefined,
  })
}

onMounted(async () => {
  fetchFiltered()
  if (auth.isAuthenticated) {
    try {
      const res = await getProgress()
      solvedSlugs.value = new Set(res.data.progress.solved_slugs ?? [])
      attemptedSlugs.value = new Set(res.data.progress.attempted_slugs ?? [])
    } catch {
      // Progress not available
    }
  }
})

watch([category, difficulty], () => {
  fetchFiltered()
})

function setCategory(c: string) {
  category.value = category.value === c ? '' : c
}

function setDifficulty(d: string) {
  difficulty.value = difficulty.value === d ? '' : d
}

function statusIcon(slug: string): string {
  if (solvedSlugs.value.has(slug)) return 'solved'
  if (attemptedSlugs.value.has(slug)) return 'attempted'
  return ''
}
</script>

<template>
  <div class="problem-list-page">
    <div class="page-header">
      <h1>Problems</h1>
      <div class="search-bar">
        <input
          v-model="search"
          type="text"
          placeholder="Search problems..."
          @keyup.enter="fetchFiltered"
        />
      </div>
    </div>

    <div class="filters">
      <div class="filter-group">
        <span class="filter-label">Category</span>
        <div class="filter-buttons">
          <button
            v-for="c in categories"
            :key="c"
            :class="['filter-btn', { active: category === c }]"
            @click="setCategory(c)"
          >
            {{ c }}
          </button>
        </div>
      </div>

      <div class="filter-group">
        <span class="filter-label">Difficulty</span>
        <div class="filter-buttons">
          <button
            v-for="d in difficulties"
            :key="d"
            :class="['filter-btn', `filter-${d}`, { active: difficulty === d }]"
            @click="setDifficulty(d)"
          >
            {{ d }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="store.loading" class="loading">Loading problems...</div>

    <div v-else-if="store.problems.length === 0" class="empty">
      No problems found. Try adjusting your filters.
    </div>

    <div v-else class="problem-grid">
      <ProblemCard
        v-for="problem in store.problems"
        :key="problem.id"
        :problem="problem"
        :status="statusIcon(problem.slug)"
      />
    </div>
  </div>
</template>

<style scoped>
.problem-list-page {
  max-width: 900px;
  margin: 0 auto;
  padding: 32px 24px;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
  gap: 16px;
}

.page-header h1 {
  font-size: 1.5rem;
  font-weight: 700;
}

.search-bar input {
  width: 280px;
}

.filters {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 24px;
}

.filter-group {
  display: flex;
  align-items: center;
  gap: 12px;
}

.filter-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  min-width: 70px;
}

.filter-buttons {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.filter-btn {
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 12px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  cursor: pointer;
  transition: all 0.2s;
  text-transform: capitalize;
  font-family: inherit;
}

.filter-btn:hover,
.filter-btn.active {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}

.filter-easy.active { background: var(--easy); border-color: var(--easy); }
.filter-medium.active { background: var(--medium); border-color: var(--medium); }
.filter-hard.active { background: var(--hard); border-color: var(--hard); }

.loading,
.empty {
  text-align: center;
  color: var(--text-secondary);
  padding: 40px;
  font-size: 14px;
}

.problem-grid {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

@media (max-width: 640px) {
  .page-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .search-bar input {
    width: 100%;
  }

  .filter-group {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>

import { defineStore } from 'pinia'
import { ref } from 'vue'
import { listProblems, getProblem } from '@/api/problems'
import type { Problem, ProblemDetail } from '@/api/problems'

export const useProblemsStore = defineStore('problems', () => {
  const problems = ref<Problem[]>([])
  const currentProblem = ref<ProblemDetail | null>(null)
  const loading = ref(false)

  async function fetchProblems(params?: { category?: string; difficulty?: string; search?: string }) {
    loading.value = true
    try {
      const res = await listProblems(params)
      problems.value = res.data.problems
    } finally {
      loading.value = false
    }
  }

  async function fetchProblem(slug: string) {
    loading.value = true
    try {
      const res = await getProblem(slug)
      currentProblem.value = res.data.problem
    } finally {
      loading.value = false
    }
  }

  return { problems, currentProblem, loading, fetchProblems, fetchProblem }
})

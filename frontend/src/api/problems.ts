import client from './client'

export interface Problem {
  id: string
  title: string
  slug: string
  category: string
  difficulty: string
  description: string
  template_code: string
  function_name: string
  hint: string
  signature: string
  example: string
  sort_order: number
}

export type ProblemDetail = Problem

export interface TestResult {
  name: string
  passed: boolean
  message: string
}

export interface SubmissionResult {
  passed: boolean
  output: string
  error: string
  test_results: TestResult[]
  execution_time?: number
}

export interface Submission {
  id: string
  problem_slug: string
  problem_title: string
  code: string
  passed: number
  total: number
  success: boolean
  execution_time: number
  created_at: string
}

export interface UserProgress {
  solved: number
  attempted: number
  total: number
  points: number
  rank: number
  by_difficulty: {
    easy: { solved: number; total: number }
    medium: { solved: number; total: number }
    hard: { solved: number; total: number }
  }
  solved_slugs: string[]
  attempted_slugs: string[]
}

export function listProblems(params?: { category?: string; difficulty?: string; search?: string }) {
  return client.get<{ problems: Problem[] }>('/problems', { params })
}

export function getProblem(slug: string) {
  return client.get<{ problem: ProblemDetail }>(`/problems/${slug}`)
}

export function submitSolution(slug: string, code: string) {
  return client.post<{ result: SubmissionResult }>(`/problems/${slug}/submit`, { code })
}

export function getSolution(slug: string) {
  return client.get<{ solution: string; solution_code: string }>(`/problems/${slug}/solution`)
}

export function getProgress() {
  return client.get<{ progress: UserProgress }>('/users/me/progress')
}

export function getSubmissions(page = 1, limit = 20) {
  return client.get<{ submissions: Submission[]; total: number }>('/users/me/submissions', {
    params: { page, limit },
  })
}

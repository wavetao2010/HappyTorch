import client from './client'

export interface CommunityProblem {
  id: string
  title: string
  slug: string
  difficulty: 'easy' | 'medium' | 'hard'
  category: string
  function_name: string
  description: string
  hint: string
  template_code: string
  solution_code: string
  solution_markdown: string
  tests: ProblemTest[]
  status: 'pending' | 'approved' | 'rejected'
  review_note: string
  created_at: string
  updated_at: string
}

export interface ProblemTest {
  name: string
  code: string
}

export interface SubmitProblemPayload {
  title: string
  difficulty: 'easy' | 'medium' | 'hard'
  category: string
  function_name: string
  description: string
  hint?: string
  template_code: string
  tests: ProblemTest[]
  solution_code?: string
  solution_markdown?: string
}

export interface LeaderboardEntry {
  rank: number
  user_id: string
  username: string
  display_name: string
  avatar_url: string
  points: number
  problems_solved: number
  contributions: number
}

export interface PublicUserProfile {
  id: string
  username: string
  display_name: string
  avatar_url: string
  bio: string
  role: string
  points: number
  problems_solved: number
  contributions: number
  created_at: string
}

export function submitProblem(data: SubmitProblemPayload) {
  return client.post<{ problem: CommunityProblem }>('/problems/submit', data)
}

export function getMyProblems() {
  return client.get<{ problems: CommunityProblem[] }>('/problems/mine')
}

export function updateProblem(id: string, data: SubmitProblemPayload) {
  return client.put<{ problem: CommunityProblem }>(`/problems/${id}`, data)
}

export function getLeaderboard(sort = 'points', limit = 50) {
  return client.get<{ entries: LeaderboardEntry[] }>('/leaderboard', {
    params: { sort, limit },
  })
}

export function getUserProfile(username: string) {
  return client.get<{ user: PublicUserProfile }>(`/users/${username}`)
}

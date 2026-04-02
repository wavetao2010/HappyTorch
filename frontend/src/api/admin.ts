import client from './client'
import type { CommunityProblem } from './community'

export interface AdminUser {
  id: string
  username: string
  email: string
  display_name: string
  role: string
  points: number
  problems_solved: number
  created_at: string
}

export function getPendingProblems() {
  return client.get<{ problems: CommunityProblem[] }>('/admin/problems/pending')
}

export function getPendingProblem(id: string) {
  return client.get<{ problem: CommunityProblem }>(`/admin/problems/${id}`)
}

export function reviewProblem(id: string, action: 'approve' | 'reject', note?: string) {
  return client.post<{ problem: CommunityProblem }>(`/admin/problems/${id}/review`, {
    action,
    note,
  })
}

export function getUsers(page = 1, limit = 20) {
  return client.get<{ users: AdminUser[]; total: number }>('/admin/users', {
    params: { page, limit },
  })
}

export function updateUserRole(id: string, role: string) {
  return client.put(`/admin/users/${id}/role`, { role })
}

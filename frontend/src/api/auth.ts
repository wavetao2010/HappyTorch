import client from './client'

export interface User {
  id: string
  username: string
  email: string
  display_name: string
  avatar_url: string
  role: string
  points: number
  problems_solved: number
  bio: string
  created_at: string
  updated_at: string
}

export interface LoginPayload {
  email: string
  password: string
}

export interface RegisterPayload {
  username: string
  email: string
  password: string
}

export function login(payload: LoginPayload) {
  return client.post<{ user: User }>('/auth/login', payload)
}

export function register(payload: RegisterPayload) {
  return client.post<{ user: User }>('/auth/register', payload)
}

export function logout() {
  return client.post('/auth/logout')
}

export function getMe() {
  return client.get<{ user: User }>('/auth/me')
}

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { getUsers, updateUserRole } from '@/api/admin'
import type { AdminUser } from '@/api/admin'

const auth = useAuthStore()
const users = ref<AdminUser[]>([])
const total = ref(0)
const page = ref(1)
const limit = 20
const loading = ref(true)

async function fetchUsers() {
  loading.value = true
  try {
    const res = await getUsers(page.value, limit)
    users.value = res.data.users ?? []
    total.value = res.data.total ?? 0
  } catch {
    // Failed
  } finally {
    loading.value = false
  }
}

async function changeRole(user: AdminUser, newRole: string) {
  if (newRole === user.role) return
  try {
    await updateUserRole(user.id, newRole)
    user.role = newRole
  } catch {
    // Failed
  }
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

const totalPages = () => Math.ceil(total.value / limit)

function goPage(p: number) {
  if (p < 1 || p > totalPages()) return
  page.value = p
  fetchUsers()
}

onMounted(fetchUsers)
</script>

<template>
  <div class="users-page">
    <h1>User Management</h1>

    <div v-if="loading" class="loading">Loading users...</div>

    <template v-else>
      <div class="users-table-wrapper">
        <table class="users-table">
          <thead>
            <tr>
              <th>Username</th>
              <th>Email</th>
              <th>Role</th>
              <th>Points</th>
              <th>Solved</th>
              <th>Joined</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="u in users" :key="u.id">
              <td class="username-cell">
                <router-link :to="`/users/${u.username}`">{{ u.username }}</router-link>
              </td>
              <td>{{ u.email }}</td>
              <td>
                <select
                  v-if="auth.isAdmin"
                  :value="u.role"
                  class="role-select"
                  @change="changeRole(u, ($event.target as HTMLSelectElement).value)"
                >
                  <option value="user">user</option>
                  <option value="moderator">moderator</option>
                  <option value="admin">admin</option>
                </select>
                <span v-else class="role-text">{{ u.role }}</span>
              </td>
              <td class="num-cell">{{ u.points }}</td>
              <td class="num-cell">{{ u.problems_solved }}</td>
              <td class="date-cell">{{ formatDate(u.created_at) }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div v-if="totalPages() > 1" class="pagination">
        <button class="btn btn-secondary btn-sm" :disabled="page <= 1" @click="goPage(page - 1)">Prev</button>
        <span class="page-info">Page {{ page }} of {{ totalPages() }}</span>
        <button class="btn btn-secondary btn-sm" :disabled="page >= totalPages()" @click="goPage(page + 1)">Next</button>
      </div>
    </template>
  </div>
</template>

<style scoped>
.users-page {
  max-width: 900px;
  margin: 0 auto;
  padding: 32px 24px;
}

.users-page h1 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 24px;
}

.loading {
  text-align: center;
  color: var(--text-secondary);
  padding: 40px;
  font-size: 14px;
}

.users-table-wrapper {
  overflow-x: auto;
}

.users-table {
  width: 100%;
  border-collapse: collapse;
}

.users-table th {
  text-align: left;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
}

.users-table td {
  padding: 12px;
  border-bottom: 1px solid var(--border);
  font-size: 14px;
}

.username-cell a {
  font-weight: 500;
}

.num-cell {
  font-family: 'JetBrains Mono', monospace;
  text-align: right;
}

.date-cell {
  color: var(--text-secondary);
  font-size: 13px;
}

.role-select {
  width: auto;
  padding: 4px 8px;
  font-size: 13px;
}

.role-text {
  text-transform: capitalize;
}

.pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  margin-top: 20px;
}

.page-info {
  font-size: 13px;
  color: var(--text-secondary);
}

.btn-sm {
  padding: 4px 10px;
  font-size: 12px;
}
</style>

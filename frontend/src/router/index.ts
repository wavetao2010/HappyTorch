import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import HomeView from '@/views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    },
    {
      path: '/login',
      name: 'login',
      component: () => import('@/views/LoginView.vue'),
      meta: { guest: true },
    },
    {
      path: '/register',
      name: 'register',
      component: () => import('@/views/RegisterView.vue'),
      meta: { guest: true },
    },
    {
      path: '/problems',
      name: 'problems',
      component: () => import('@/views/ProblemListView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/problems/submit',
      name: 'problem-submit',
      component: () => import('@/views/ProblemSubmitView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/problems/mine',
      name: 'my-problems',
      component: () => import('@/views/MyProblemsView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/problems/:slug',
      name: 'problem-detail',
      component: () => import('@/views/ProblemDetailView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/submissions',
      name: 'submissions',
      component: () => import('@/views/SubmissionHistoryView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/profile',
      name: 'profile',
      component: () => import('@/views/ProfileView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/leaderboard',
      name: 'leaderboard',
      component: () => import('@/views/LeaderboardView.vue'),
    },
    {
      path: '/users/:username',
      name: 'user-profile',
      component: () => import('@/views/UserProfileView.vue'),
    },
    {
      path: '/admin/review',
      name: 'admin-review',
      component: () => import('@/views/AdminReviewView.vue'),
      meta: { requiresAuth: true, requiresModerator: true },
    },
    {
      path: '/admin/users',
      name: 'admin-users',
      component: () => import('@/views/AdminUsersView.vue'),
      meta: { requiresAuth: true, requiresAdmin: true },
    },
  ],
})

router.beforeEach(async (to) => {
  const auth = useAuthStore()

  if (!auth.isAuthenticated && !auth.loading) {
    await auth.fetchUser()
  }

  if (to.meta.requiresAuth && !auth.isAuthenticated) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }

  if (to.meta.guest && auth.isAuthenticated) {
    return { name: 'problems' }
  }

  if (to.meta.requiresModerator && !auth.isModerator) {
    return { name: 'problems' }
  }

  if (to.meta.requiresAdmin && !auth.isAdmin) {
    return { name: 'problems' }
  }
})

export default router

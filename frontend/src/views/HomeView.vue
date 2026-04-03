<script setup lang="ts">
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
</script>

<template>
  <div class="home">
    <div class="hero">
      <h1 class="hero-title">HappyTorch</h1>
      <p class="hero-subtitle">Master PyTorch through hands-on coding challenges</p>
      <p class="hero-desc">A free, open-source community platform for learning PyTorch.</p>
      <div class="hero-actions">
        <router-link v-if="auth.isAuthenticated" to="/problems" class="btn btn-primary btn-lg">
          Browse Problems
        </router-link>
        <template v-else>
          <router-link to="/register" class="btn btn-primary btn-lg">Get Started</router-link>
          <router-link to="/login" class="btn btn-secondary btn-lg">Login</router-link>
        </template>
      </div>
      <a href="https://github.com/Rivflyyy/HappyTorch" target="_blank" class="github-star">
        <svg class="github-icon" viewBox="0 0 16 16" width="18" height="18" fill="currentColor">
          <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
        </svg>
        Star on GitHub
      </a>
    </div>

    <div class="features">
      <div class="feature-card">
        <div class="feature-icon">&#x1F9E0;</div>
        <h3>Learn by Doing</h3>
        <p>Solve real PyTorch problems from basic tensor ops to advanced neural networks.</p>
      </div>
      <div class="feature-card">
        <div class="feature-icon">&#x26A1;</div>
        <h3>Instant Feedback</h3>
        <p>Submit your solutions and get immediate test results with detailed output.</p>
      </div>
      <div class="feature-card">
        <div class="feature-icon">&#x1F4C8;</div>
        <h3>Track Progress</h3>
        <p>Monitor your growth across categories and difficulty levels.</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.home {
  max-width: 900px;
  margin: 0 auto;
  padding: 60px 24px;
}

.hero {
  text-align: center;
  margin-bottom: 60px;
}

.hero-title {
  font-size: 3rem;
  font-weight: 700;
  margin-bottom: 12px;
  background: linear-gradient(135deg, #ff6b35, #f7931a);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-subtitle {
  font-size: 1.2rem;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

.hero-desc {
  font-size: 1rem;
  color: var(--text-muted);
  margin-bottom: 32px;
}

.github-star {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 24px;
  padding: 10px 20px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text-primary);
  font-size: 14px;
  font-weight: 500;
  text-decoration: none;
  transition: all 0.2s;
}

.github-star:hover {
  background: var(--bg-tertiary);
  border-color: var(--accent);
}

.github-icon {
  flex-shrink: 0;
}

.hero-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
}

.btn-lg {
  padding: 12px 28px;
  font-size: 16px;
}

.features {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.feature-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 28px 24px;
  text-align: center;
}

.feature-icon {
  font-size: 2rem;
  margin-bottom: 12px;
}

.feature-card h3 {
  font-size: 1rem;
  margin-bottom: 8px;
}

.feature-card p {
  font-size: 0.875rem;
  color: var(--text-secondary);
  line-height: 1.5;
}

@media (max-width: 640px) {
  .features {
    grid-template-columns: 1fr;
  }

  .hero-title {
    font-size: 2rem;
  }
}
</style>

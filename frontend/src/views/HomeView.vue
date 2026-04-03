<script setup lang="ts">
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
</script>

<template>
  <div class="home">
    <div class="hero">
      <h1 class="hero-title">HappyTorch</h1>
      <p class="hero-subtitle">Master PyTorch through hands-on coding challenges</p>
      <div class="hero-actions">
        <router-link v-if="auth.isAuthenticated" to="/problems" class="btn btn-primary btn-lg">
          Browse Problems
        </router-link>
        <template v-else>
          <router-link to="/register" class="btn btn-primary btn-lg">Get Started</router-link>
          <router-link to="/login" class="btn btn-secondary btn-lg">Login</router-link>
        </template>
      </div>
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
  margin-bottom: 32px;
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

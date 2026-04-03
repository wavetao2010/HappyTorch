<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import MonacoEditor from '@/components/MonacoEditor.vue'
import { submitProblem } from '@/api/community'
import type { ProblemTest } from '@/api/community'

const router = useRouter()
const submitting = ref(false)
const error = ref('')
const success = ref(false)

const form = reactive({
  title: '',
  difficulty: 'easy' as 'easy' | 'medium' | 'hard',
  category: '',
  function_name: '',
  description: '',
  hint: '',
  template_code: 'import torch\n\ndef solution():\n    pass\n',
  solution_code: '',
  solution_markdown: '',
  tests: [{ name: 'Test 1', code: '' }] as ProblemTest[],
})

function addTest() {
  form.tests.push({ name: `Test ${form.tests.length + 1}`, code: '' })
}

function removeTest(index: number) {
  if (form.tests.length > 1) {
    form.tests.splice(index, 1)
  }
}

async function handleSubmit() {
  error.value = ''
  if (!form.title || !form.category || !form.function_name || !form.description || !form.template_code) {
    error.value = 'Please fill in all required fields.'
    return
  }
  if (form.tests.some((t) => !t.name || !t.code)) {
    error.value = 'All tests must have a name and code.'
    return
  }

  submitting.value = true
  try {
    await submitProblem({
      title: form.title,
      difficulty: form.difficulty,
      category: form.category,
      function_name: form.function_name,
      description: form.description,
      hint: form.hint || undefined,
      template_code: form.template_code,
      tests: form.tests,
      solution_code: form.solution_code || undefined,
      solution_markdown: form.solution_markdown || undefined,
    })
    success.value = true
    setTimeout(() => router.push('/problems/mine'), 1500)
  } catch (err: unknown) {
    const msg = (err as { response?: { data?: { error?: string } } })?.response?.data?.error
    error.value = msg || 'Failed to submit problem.'
  } finally {
    submitting.value = false
  }
}

const categories = ['Tensor Basics', 'Autograd', 'Neural Networks', 'Data Loading', 'Training', 'Advanced']
</script>

<template>
  <div class="submit-page">
    <h1>Submit a Problem</h1>
    <p class="subtitle">Create a new PyTorch problem for the community. It will be reviewed by moderators before appearing.</p>

    <div v-if="success" class="success-msg">Problem submitted successfully! Redirecting...</div>

    <form v-else @submit.prevent="handleSubmit" class="submit-form">
      <div v-if="error" class="error-msg">{{ error }}</div>

      <div class="form-row">
        <div class="form-group flex-1">
          <label>Title *</label>
          <input v-model="form.title" type="text" placeholder="e.g. Reshape a Tensor" />
        </div>
        <div class="form-group">
          <label>Difficulty *</label>
          <select v-model="form.difficulty">
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>
        </div>
      </div>

      <div class="form-row">
        <div class="form-group flex-1">
          <label>Category *</label>
          <select v-model="form.category">
            <option value="" disabled>Select a category</option>
            <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
          </select>
        </div>
        <div class="form-group flex-1">
          <label>Function Name *</label>
          <input v-model="form.function_name" type="text" placeholder="e.g. reshape_tensor" />
        </div>
      </div>

      <div class="form-group">
        <label>Description (Markdown) *</label>
        <MonacoEditor v-model="form.description" language="markdown" height="200px" />
      </div>

      <div class="form-group">
        <label>Hint</label>
        <input v-model="form.hint" type="text" placeholder="Optional hint for the problem" />
      </div>

      <div class="form-group">
        <label>Template Code (Python) *</label>
        <MonacoEditor v-model="form.template_code" language="python" height="200px" />
      </div>

      <div class="form-group">
        <div class="tests-header">
          <label>Tests *</label>
          <button type="button" class="btn btn-secondary btn-sm" @click="addTest">+ Add Test</button>
        </div>
        <div v-for="(test, i) in form.tests" :key="i" class="test-item">
          <div class="test-header">
            <input v-model="test.name" type="text" placeholder="Test name" class="test-name-input" />
            <button
              v-if="form.tests.length > 1"
              type="button"
              class="btn-remove"
              @click="removeTest(i)"
            >Remove</button>
          </div>
          <MonacoEditor v-model="test.code" language="python" height="120px" />
        </div>
      </div>

      <div class="form-group">
        <label>Solution Code (optional)</label>
        <MonacoEditor v-model="form.solution_code" language="python" height="200px" />
      </div>

      <div class="form-group">
        <label>Solution Explanation (optional, Markdown)</label>
        <MonacoEditor v-model="form.solution_markdown" language="markdown" height="150px" />
      </div>

      <button type="submit" class="btn btn-primary" :disabled="submitting">
        {{ submitting ? 'Submitting...' : 'Submit Problem' }}
      </button>
    </form>
  </div>
</template>

<style scoped>
.submit-page {
  max-width: 800px;
  margin: 0 auto;
  padding: 32px 24px;
}

.submit-page h1 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 8px;
}

.subtitle {
  color: var(--text-secondary);
  font-size: 14px;
  margin-bottom: 24px;
}

.submit-form {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.form-row {
  display: flex;
  gap: 16px;
}

.flex-1 {
  flex: 1;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.form-group label {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
}

.error-msg {
  background: rgba(248, 81, 73, 0.15);
  color: var(--error);
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 14px;
}

.success-msg {
  background: rgba(63, 185, 80, 0.15);
  color: var(--success);
  padding: 16px;
  border-radius: 8px;
  font-size: 14px;
  text-align: center;
}

.tests-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.test-item {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  margin-top: 8px;
}

.test-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

.test-name-input {
  flex: 1;
  padding: 6px 10px;
  font-size: 13px;
}

.btn-remove {
  background: none;
  border: none;
  color: var(--error);
  cursor: pointer;
  font-size: 13px;
  font-family: inherit;
}

.btn-sm {
  padding: 4px 10px;
  font-size: 12px;
}

@media (max-width: 640px) {
  .form-row {
    flex-direction: column;
  }
}
</style>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useProblemsStore } from '@/stores/problems'
import { submitSolution, getSolution } from '@/api/problems'
import type { SubmissionResult } from '@/api/problems'
import MonacoEditor from '@/components/MonacoEditor.vue'
import { marked } from 'marked'
import katex from 'katex'

const route = useRoute()
const store = useProblemsStore()

const code = ref('')
const activeTab = ref<'description' | 'solution'>('description')
const submitting = ref(false)
const result = ref<SubmissionResult | null>(null)
const solutionMarkdown = ref('')
const solutionLoaded = ref(false)
const showResults = ref(false)

const slug = computed(() => route.params.slug as string)

function renderMarkdown(md: string): string {
  const html = marked.parse(md, { async: false }) as string
  return html.replace(/\$\$([\s\S]+?)\$\$/g, (_match, tex) => {
    try {
      return katex.renderToString(tex.trim(), { displayMode: true, throwOnError: false })
    } catch {
      return tex
    }
  }).replace(/\$([\s\S]+?)\$/g, (_match, tex) => {
    try {
      return katex.renderToString(tex.trim(), { displayMode: false, throwOnError: false })
    } catch {
      return tex
    }
  })
}

const descriptionHtml = computed(() => {
  if (!store.currentProblem) return ''
  return renderMarkdown(store.currentProblem.description)
})

const solutionHtml = computed(() => {
  if (!solutionMarkdown.value) return ''
  return renderMarkdown(solutionMarkdown.value)
})

const allPassed = computed(() => {
  if (!result.value) return false
  return result.value.test_results.every((t) => t.passed)
})

onMounted(async () => {
  await store.fetchProblem(slug.value)
  if (store.currentProblem) {
    code.value = store.currentProblem.template_code || ''
  }
})

async function handleSubmit() {
  submitting.value = true
  result.value = null
  showResults.value = true
  try {
    const res = await submitSolution(slug.value, code.value)
    result.value = res.data.result
  } catch (err: unknown) {
    const msg = (err as { response?: { data?: { error?: string } } })?.response?.data?.error
    result.value = {
      passed: false,
      output: '',
      error: msg || 'Submission failed. Please try again.',
      test_results: [],
    }
  } finally {
    submitting.value = false
  }
}

async function loadSolution() {
  activeTab.value = 'solution'
  if (solutionLoaded.value) return
  try {
    const res = await getSolution(slug.value)
    solutionMarkdown.value = res.data.solution
    solutionLoaded.value = true
  } catch {
    solutionMarkdown.value = 'Solution not available yet.'
    solutionLoaded.value = true
  }
}

</script>

<template>
  <div class="problem-detail-page">
    <div v-if="store.loading" class="loading">Loading problem...</div>

    <template v-else-if="store.currentProblem">
      <div class="problem-header">
        <div class="problem-title-row">
          <h1>{{ store.currentProblem.title }}</h1>
          <span :class="['difficulty-badge', `difficulty-${store.currentProblem.difficulty.toLowerCase()}`]">
            {{ store.currentProblem.difficulty }}
          </span>
          <span class="category-badge">{{ store.currentProblem.category }}</span>
        </div>
        <div class="problem-tabs">
          <button
            :class="['tab-btn', { active: activeTab === 'description' }]"
            @click="activeTab = 'description'"
          >
            Description
          </button>
          <button
            :class="['tab-btn', { active: activeTab === 'solution' }]"
            @click="loadSolution"
          >
            Solution
          </button>
        </div>
      </div>

      <div class="problem-body">
        <!-- Left panel -->
        <div class="left-panel">
          <div v-if="activeTab === 'description'" class="description-panel">
            <div class="markdown-content" v-html="descriptionHtml" />

            <div v-if="store.currentProblem.signature" class="section">
              <h3>Function Signature</h3>
              <pre class="code-block">{{ store.currentProblem.signature }}</pre>
            </div>

            <div
              v-if="store.currentProblem.hint"
              class="section"
            >
              <h3>Hint</h3>
              <div
                class="markdown-content hints-content"
                v-html="renderMarkdown(store.currentProblem.hint)"
              />
            </div>
          </div>

          <div v-else class="solution-panel">
            <div v-if="!solutionLoaded" class="loading">Loading solution...</div>
            <div v-else class="markdown-content" v-html="solutionHtml" />
          </div>
        </div>

        <!-- Right panel -->
        <div class="right-panel">
          <div class="editor-header">
            <span class="editor-label">Python</span>
            <button
              class="btn btn-primary submit-btn"
              :disabled="submitting"
              @click="handleSubmit"
            >
              <template v-if="submitting">Running...</template>
              <template v-else>Submit</template>
            </button>
          </div>

          <MonacoEditor v-model="code" language="python" height="400px" />

          <!-- Results panel -->
          <div v-if="showResults" class="results-panel">
            <div class="results-header">
              <h3>Results</h3>
              <span v-if="result && !submitting" :class="['result-status', { passed: allPassed, failed: !allPassed }]">
                {{ allPassed ? 'All Passed' : 'Some Failed' }}
              </span>
            </div>

            <div v-if="submitting" class="results-loading">Running tests...</div>

            <template v-else-if="result">
              <div v-if="result.error" class="error-output">
                <pre>{{ result.error }}</pre>
              </div>

              <div v-if="result.test_results.length > 0" class="test-results">
                <div
                  v-for="(test, i) in result.test_results"
                  :key="i"
                  :class="['test-item', { passed: test.passed, failed: !test.passed }]"
                >
                  <span class="test-icon">{{ test.passed ? 'PASS' : 'FAIL' }}</span>
                  <span class="test-name">{{ test.name }}</span>
                  <span v-if="test.message" class="test-message">{{ test.message }}</span>
                </div>
              </div>

              <div v-if="result.output" class="stdout-output">
                <h4>Output</h4>
                <pre>{{ result.output }}</pre>
              </div>

              <div v-if="result.execution_time" class="execution-time">
                Execution time: {{ result.execution_time }}ms
              </div>
            </template>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.problem-detail-page {
  height: calc(100vh - 56px);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.loading {
  text-align: center;
  color: var(--text-secondary);
  padding: 40px;
}

.problem-header {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.problem-title-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 24px 8px;
}

.problem-title-row h1 {
  font-size: 1.25rem;
  font-weight: 600;
}

.difficulty-badge {
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  text-transform: capitalize;
}

.difficulty-easy {
  background: rgba(63, 185, 80, 0.15);
  color: var(--easy);
}

.difficulty-medium {
  background: rgba(210, 153, 34, 0.15);
  color: var(--medium);
}

.difficulty-hard {
  background: rgba(248, 81, 73, 0.15);
  color: var(--hard);
}

.category-badge {
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 12px;
  background: var(--bg-tertiary);
  color: var(--text-secondary);
}

.problem-tabs {
  display: flex;
  gap: 4px;
  padding: 0 24px;
}

.tab-btn {
  padding: 12px 20px;
  font-size: 14px;
  background: transparent;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  transition: all 0.2s;
  border-bottom: 2px solid transparent;
  font-family: inherit;
}

.tab-btn:hover {
  color: var(--text-primary);
}

.tab-btn.active {
  color: var(--text-primary);
  border-bottom-color: var(--accent);
}

.problem-body {
  display: flex;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.left-panel {
  width: 45%;
  overflow-y: auto;
  padding: 24px;
  border-right: 1px solid var(--border);
}

.right-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  padding: 16px;
}

.editor-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.editor-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.submit-btn {
  min-width: 100px;
  justify-content: center;
}

.submit-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.section {
  margin-top: 24px;
}

.section h3 {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 10px;
  color: var(--text-primary);
}

.code-block {
  padding: 16px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: var(--text-primary);
  overflow-x: auto;
  white-space: pre;
}

/* Results */
.results-panel {
  margin-top: 16px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}

.results-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.results-header h3 {
  font-size: 14px;
  font-weight: 600;
}

.result-status {
  font-size: 13px;
  font-weight: 600;
}

.result-status.passed {
  color: var(--success);
}

.result-status.failed {
  color: var(--error);
}

.results-loading {
  color: var(--text-secondary);
  font-size: 14px;
  padding: 12px 0;
}

.error-output {
  margin-bottom: 12px;
}

.error-output pre {
  padding: 12px;
  background: rgba(248, 81, 73, 0.1);
  border: 1px solid rgba(248, 81, 73, 0.3);
  border-radius: 6px;
  color: var(--error);
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  white-space: pre-wrap;
  word-break: break-word;
}

.test-results {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.test-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 13px;
}

.test-item.passed {
  background: rgba(63, 185, 80, 0.08);
}

.test-item.failed {
  background: rgba(248, 81, 73, 0.08);
}

.test-icon {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 700;
  min-width: 36px;
}

.test-item.passed .test-icon {
  color: var(--success);
}

.test-item.failed .test-icon {
  color: var(--error);
}

.test-name {
  font-weight: 500;
  color: var(--text-primary);
}

.test-message {
  color: var(--text-secondary);
  font-size: 12px;
  margin-left: auto;
}

.stdout-output {
  margin-top: 12px;
}

.stdout-output h4 {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 6px;
}

.stdout-output pre {
  padding: 12px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: var(--text-primary);
  white-space: pre-wrap;
}

.execution-time {
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-muted);
}

/* Markdown content styles */
.markdown-content :deep(h2) {
  font-size: 1.1rem;
  margin-bottom: 16px;
  color: var(--text-primary);
}

.markdown-content :deep(h3) {
  font-size: 1rem;
  margin-top: 20px;
  margin-bottom: 10px;
  color: var(--text-primary);
}

.markdown-content :deep(p) {
  margin-bottom: 12px;
  line-height: 1.7;
  font-size: 14px;
  color: var(--text-primary);
}

.markdown-content :deep(code) {
  background: var(--bg-tertiary);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
}

.markdown-content :deep(pre) {
  background: var(--bg-tertiary);
  padding: 16px;
  border-radius: 8px;
  overflow-x: auto;
  margin-bottom: 12px;
}

.markdown-content :deep(pre code) {
  padding: 0;
  background: transparent;
}

.markdown-content :deep(ul),
.markdown-content :deep(ol) {
  margin-bottom: 12px;
  padding-left: 24px;
}

.markdown-content :deep(li) {
  margin-bottom: 4px;
  line-height: 1.6;
  font-size: 14px;
}

.markdown-content :deep(.katex-display) {
  margin: 16px 0;
  overflow-x: auto;
}

@media (max-width: 900px) {
  .problem-body {
    flex-direction: column;
  }

  .left-panel {
    width: 100%;
    border-right: none;
    border-bottom: 1px solid var(--border);
    max-height: 40vh;
  }
}
</style>

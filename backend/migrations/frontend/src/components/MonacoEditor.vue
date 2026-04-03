<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch, shallowRef } from 'vue'
import type * as Monaco from 'monaco-editor'
import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker'

self.MonacoEnvironment = {
  getWorker: () => new editorWorker(),
}

const props = withDefaults(
  defineProps<{
    modelValue: string
    language?: string
    theme?: string
    height?: string
    readOnly?: boolean
  }>(),
  {
    language: 'python',
    theme: 'vs-dark',
    height: '400px',
    readOnly: false,
  },
)

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

const container = ref<HTMLDivElement>()
const editor = shallowRef<Monaco.editor.IStandaloneCodeEditor>()
let monaco: typeof Monaco

onMounted(async () => {
  monaco = await import('monaco-editor')

  if (!container.value) return

  editor.value = monaco.editor.create(container.value, {
    value: props.modelValue,
    language: props.language,
    theme: props.theme,
    readOnly: props.readOnly,
    minimap: { enabled: false },
    fontSize: 14,
    fontFamily: "'JetBrains Mono', monospace",
    lineNumbers: 'on',
    scrollBeyondLastLine: false,
    automaticLayout: true,
    tabSize: 4,
    insertSpaces: true,
    padding: { top: 12 },
    renderLineHighlight: 'line',
    cursorBlinking: 'smooth',
  })

  editor.value.onDidChangeModelContent(() => {
    const value = editor.value!.getValue()
    emit('update:modelValue', value)
  })
})

watch(
  () => props.modelValue,
  (newVal) => {
    if (editor.value && editor.value.getValue() !== newVal) {
      editor.value.setValue(newVal)
    }
  },
)

watch(
  () => props.readOnly,
  (val) => {
    editor.value?.updateOptions({ readOnly: val })
  },
)

onBeforeUnmount(() => {
  editor.value?.dispose()
})
</script>

<template>
  <div ref="container" class="monaco-container" :style="{ height }" />
</template>

<style scoped>
.monaco-container {
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}
</style>

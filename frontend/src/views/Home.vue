<template>
  <div class="home">
    <!-- Top navigation -->
    <nav class="nav">
      <div class="nav-inner">
        <div class="nav-brand">
          <span class="nav-brand-mark">⬢</span>
          <span class="nav-brand-name">MiroFish</span>
        </div>
        <div class="nav-links">
          <LanguageSwitcher />
          <a
            href="https://github.com/666ghj/MiroFish"
            target="_blank"
            class="nav-external"
          >
            {{ $t('nav.visitGithub') }} <span class="nav-external-arrow">↗</span>
          </a>
        </div>
      </div>
    </nav>

    <main class="content">
      <!-- ──────── Hero ──────── -->
      <section class="hero">
        <div class="hero-left">
          <div class="hero-eyebrow">
            <span class="hero-eyebrow-text">{{ $t('home.eyebrow') }}</span>
            <span class="hero-eyebrow-sep"></span>
            <span class="hero-eyebrow-version">{{ $t('home.version') }}</span>
          </div>

          <h1 class="hero-title">
            <span class="hero-title-line">{{ $t('home.heroTitle1') }}</span>
            <span class="hero-title-line">{{ $t('home.heroTitle2') }}</span>
            <span class="hero-title-line hero-title-accent">{{ $t('home.heroTitle3') }}</span>
          </h1>

          <p class="hero-desc">
            <i18n-t keypath="home.heroDesc" tag="span">
              <template #brand>
                <span class="hero-desc-brand">{{ $t('home.heroDescBrand') }}</span>
              </template>
              <template #optimalSolution>
                <span class="hero-desc-signal">{{ $t('home.heroDescOptimalSolution') }}</span>
              </template>
            </i18n-t>
          </p>

          <div class="hero-cta-row">
            <button class="cta-primary" @click="scrollToDeploy">
              {{ $t('home.heroPrimaryCta') }}
              <span class="cta-primary-arrow">↓</span>
            </button>
            <button class="cta-secondary" @click="scrollToMethodology">
              {{ $t('home.heroSecondaryCta') }}
            </button>
          </div>
        </div>

        <!-- ── Signature: Sentiment Waveform ── -->
        <div class="hero-right">
          <div class="waveform" aria-hidden="true">
            <div class="waveform-bars">
              <div
                v-for="(h, i) in waveformHeights"
                :key="i"
                class="waveform-bar"
                :class="{ 'waveform-bar-signal': i >= waveformSignalStart }"
                :style="{ '--bar-h': h + '%', '--bar-i': i }"
              ></div>
            </div>
            <div class="waveform-labels">
              <span class="waveform-label waveform-label-noise">
                {{ $t('home.waveformLabelNoise') }}
              </span>
              <span class="waveform-label waveform-label-signal">
                {{ $t('home.waveformLabelSignal') }}
              </span>
            </div>
          </div>
        </div>
      </section>

      <!-- ──────── Methodology ──────── -->
      <section ref="methodologySection" class="methodology">
        <div class="section-label">
          <span class="section-label-text">{{ $t('home.methodologyLabel') }}</span>
          <span class="section-label-rule"></span>
        </div>
        <h2 class="section-title">{{ $t('home.methodologyTitle') }}</h2>
        <p class="section-desc">{{ $t('home.methodologyDesc') }}</p>

        <div class="phases">
          <div
            v-for="step in phases"
            :key="step.num"
            class="phase"
          >
            <div class="phase-marker">
              <span class="phase-num">{{ step.num }}</span>
              <span
                v-if="step.num !== '05'"
                class="phase-line"
              ></span>
            </div>
            <div class="phase-body">
              <div class="phase-title">{{ $t(step.titleKey) }}</div>
              <div class="phase-desc">{{ $t(step.descKey) }}</div>
            </div>
          </div>
        </div>
      </section>

      <!-- ──────── Deploy ──────── -->
      <section ref="deploySection" class="deploy">
        <div class="section-label">
          <span class="section-label-text">{{ $t('home.deployLabel') }}</span>
          <span class="section-label-rule"></span>
        </div>
        <h2 class="section-title">{{ $t('home.deployTitle') }}</h2>
        <p class="section-desc">{{ $t('home.deployDesc') }}</p>

        <div class="deploy-grid">
          <!-- Source Documents -->
          <div class="deploy-card">
            <div class="deploy-card-header">
              <span class="deploy-card-label">{{ $t('home.realitySeed') }}</span>
              <span class="deploy-card-meta">{{ $t('home.supportedFormats') }}</span>
            </div>
            <div
              class="upload-zone"
              :class="{
                'upload-zone--drag': isDragOver,
                'upload-zone--filled': files.length > 0,
              }"
              @dragover.prevent="handleDragOver"
              @dragleave.prevent="handleDragLeave"
              @drop.prevent="handleDrop"
              @click="triggerFileInput"
              role="button"
              tabindex="0"
              @keydown.enter.prevent="triggerFileInput"
            >
              <input
                ref="fileInput"
                type="file"
                multiple
                accept=".pdf,.md,.txt,.json"
                class="upload-input"
                @change="handleFileSelect"
                :disabled="loading"
              />
              <div v-if="files.length === 0" class="upload-empty">
                <div class="upload-empty-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M12 16V4M12 4L7 9M12 4l5 5M5 20h14" stroke-linecap="round" stroke-linejoin="round"/>
                  </svg>
                </div>
                <div class="upload-empty-text">{{ $t('home.dragToUpload') }}</div>
                <div class="upload-empty-hint">{{ $t('home.orBrowse') }}</div>
              </div>
              <div v-else class="file-list">
                <div
                  v-for="(file, index) in files"
                  :key="index"
                  class="file-item"
                >
                  <span class="file-item-icon">▣</span>
                  <span class="file-item-name">{{ file.name }}</span>
                  <button
                    class="file-item-remove"
                    @click.stop="removeFile(index)"
                    :title="$t('home.removeFile')"
                    :aria-label="$t('home.removeFile')"
                  >
                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5">
                      <path d="M2 2l8 8M10 2l-8 8" stroke-linecap="round"/>
                    </svg>
                  </button>
                </div>
              </div>
            </div>
            <div v-if="files.length > 0" class="upload-count">
              {{ $t('home.fileCount', { count: files.length }) }}
            </div>
          </div>

          <!-- Assessment Brief -->
          <div class="deploy-card">
            <div class="deploy-card-header">
              <span class="deploy-card-label">{{ $t('home.simulationPrompt') }}</span>
              <span class="deploy-card-meta">{{ $t('home.engineBadge') }}</span>
            </div>
            <div class="textarea-wrapper">
              <textarea
                v-model="formData.simulationRequirement"
                class="textarea"
                :placeholder="$t('home.promptPlaceholder')"
                rows="6"
                :disabled="loading"
              ></textarea>
            </div>
          </div>
        </div>

        <button
          class="deploy-btn"
          @click="startSimulation"
          :disabled="!canSubmit || loading"
        >
          <span v-if="!loading">{{ $t('home.startEngine') }}</span>
          <span v-else>{{ $t('home.initializing') }}</span>
          <span class="deploy-btn-arrow">→</span>
        </button>
      </section>

      <!-- ──────── History ──────── -->
      <HistoryDatabase />
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import HistoryDatabase from '../components/HistoryDatabase.vue'
import LanguageSwitcher from '../components/LanguageSwitcher.vue'

const router = useRouter()

const formData = ref({
  simulationRequirement: '',
})

const files = ref([])
const loading = ref(false)
const isDragOver = ref(false)
const fileInput = ref(null)
const deploySection = ref(null)
const methodologySection = ref(null)

const phases = [
  { num: '01', titleKey: 'home.step01Title', descKey: 'home.step01Desc' },
  { num: '02', titleKey: 'home.step02Title', descKey: 'home.step02Desc' },
  { num: '03', titleKey: 'home.step03Title', descKey: 'home.step03Desc' },
  { num: '04', titleKey: 'home.step04Title', descKey: 'home.step04Desc' },
  { num: '05', titleKey: 'home.step05Title', descKey: 'home.step05Desc' },
]

const WAVEFORM_BARS = 56
const waveformSignalStart = 36

const waveformHeights = computed(() => {
  const bars = []
  for (let i = 0; i < WAVEFORM_BARS; i++) {
    if (i < waveformSignalStart) {
      const seed = (i * 7 + 13) % 17
      bars.push(20 + (seed * 41 % 55))
    } else {
      const phase = (i - waveformSignalStart) / (WAVEFORM_BARS - waveformSignalStart - 1)
      const sine = Math.sin(phase * Math.PI * 2.5) * 35 + 55
      bars.push(Math.round(sine))
    }
  }
  return bars
})

const canSubmit = computed(() => {
  return formData.value.simulationRequirement.trim() !== '' && files.value.length > 0
})

const triggerFileInput = () => {
  if (!loading.value) {
    fileInput.value?.click()
  }
}

const handleFileSelect = (event) => {
  addFiles(Array.from(event.target.files))
}

const handleDragOver = () => {
  if (!loading.value) {
    isDragOver.value = true
  }
}

const handleDragLeave = () => {
  isDragOver.value = false
}

const handleDrop = (e) => {
  isDragOver.value = false
  if (loading.value) return
  addFiles(Array.from(e.dataTransfer.files))
}

const addFiles = (newFiles) => {
  const validFiles = newFiles.filter((file) => {
    const ext = file.name.split('.').pop().toLowerCase()
    // A Run Contract may also be plain JSON (see backend services/run_contract.py)
    return ['pdf', 'md', 'txt'].includes(ext) || (ext === 'json' && /run[-_ ]?contract/i.test(file.name))
  })
  files.value.push(...validFiles)
}

const removeFile = (index) => {
  files.value.splice(index, 1)
}

const scrollToDeploy = () => {
  deploySection.value?.scrollIntoView({ behavior: 'smooth' })
}

const scrollToMethodology = () => {
  methodologySection.value?.scrollIntoView({ behavior: 'smooth' })
}

const startSimulation = () => {
  if (!canSubmit.value || loading.value) return
  import('../store/pendingUpload.js').then(({ setPendingUpload }) => {
    setPendingUpload(files.value, formData.value.simulationRequirement)
    router.push({
      name: 'Process',
      params: { projectId: 'new' },
    })
  })
}

onMounted(() => {
  document.documentElement.lang = localStorage.getItem('locale') || 'en'
})
</script>

<style scoped>
/* ════════════════════════════════════════════════════════════
   MiroFish — Depth Signal palette
   ════════════════════════════════════════════════════════════ */
.home {
  --ink-deep: #0A1B2A;
  --ink-mid: #1E3247;
  --paper: #EBEDF0;
  --surface: #F5F6F8;
  --slate: #56697D;
  --signal: #C4751A;
  --hairline: #C8CDD5;
  --white: #FFFFFF;

  --font-display: 'Fraunces', 'Noto Sans SC', Georgia, serif;
  --font-body: 'Inter', 'Noto Sans SC', system-ui, -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', monospace;

  min-height: 100vh;
  background: var(--paper);
  color: var(--ink-deep);
  font-family: var(--font-body);
  line-height: 1.6;
}

/* ── Navigation ── */
.nav {
  background: var(--ink-deep);
  color: var(--white);
  position: sticky;
  top: 0;
  z-index: 100;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.nav-inner {
  max-width: 1280px;
  margin: 0 auto;
  height: 56px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 32px;
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.nav-brand-mark {
  color: var(--signal);
  font-size: 1.1rem;
  line-height: 1;
}

.nav-brand-name {
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 1rem;
  letter-spacing: 1px;
}

.nav-links {
  display: flex;
  align-items: center;
  gap: 20px;
}

.nav-external {
  color: rgba(255, 255, 255, 0.7);
  text-decoration: none;
  font-size: 0.82rem;
  font-weight: 400;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: color 0.2s;
}

.nav-external:hover {
  color: var(--white);
}

.nav-external-arrow {
  font-family: sans-serif;
  font-size: 0.85rem;
}

/* ── Content container ── */
.content {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 32px;
}

/* ════════════════════════════════════════════════════════════
   Hero
   ════════════════════════════════════════════════════════════ */
.hero {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 64px;
  align-items: center;
  padding: 80px 0 100px;
}

.hero-eyebrow {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 28px;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 2px;
  text-transform: uppercase;
}

.hero-eyebrow-text {
  color: var(--signal);
  font-weight: 600;
}

.hero-eyebrow-sep {
  width: 24px;
  height: 1px;
  background: var(--hairline);
}

.hero-eyebrow-version {
  color: var(--slate);
  font-weight: 400;
}

.hero-title {
  font-family: var(--font-display);
  font-weight: 300;
  font-size: clamp(2.5rem, 5vw, 4.2rem);
  line-height: 1.1;
  letter-spacing: -1.5px;
  margin: 0 0 28px 0;
  color: var(--ink-deep);
}

.hero-title-line {
  display: block;
}

.hero-title-accent {
  font-style: italic;
  color: var(--signal);
}

.hero-desc {
  font-size: 1.05rem;
  line-height: 1.7;
  color: var(--ink-mid);
  max-width: 520px;
  margin-bottom: 40px;
}

.hero-desc-brand {
  font-weight: 600;
  color: var(--ink-deep);
}

.hero-desc-signal {
  color: var(--signal);
  font-weight: 500;
  border-bottom: 1px solid var(--signal);
  padding-bottom: 1px;
}

.hero-cta-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.cta-primary {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  background: var(--ink-deep);
  color: var(--white);
  border: 1px solid var(--ink-deep);
  padding: 14px 28px;
  font-family: var(--font-body);
  font-weight: 500;
  font-size: 0.95rem;
  cursor: pointer;
  transition: background 0.2s, border-color 0.2s, transform 0.15s;
  border-radius: 2px;
}

.cta-primary:hover {
  background: var(--ink-mid);
}

.cta-primary:active {
  transform: translateY(1px);
}

.cta-primary-arrow {
  font-size: 1.1rem;
  line-height: 1;
}

.cta-secondary {
  display: inline-flex;
  align-items: center;
  background: transparent;
  color: var(--ink-deep);
  border: 1px solid var(--hairline);
  padding: 14px 28px;
  font-family: var(--font-body);
  font-weight: 500;
  font-size: 0.95rem;
  cursor: pointer;
  transition: border-color 0.2s, color 0.2s;
  border-radius: 2px;
}

.cta-secondary:hover {
  border-color: var(--ink-deep);
}

/* ── Sentiment Waveform (signature element) ── */
.hero-right {
  display: flex;
  justify-content: center;
  align-items: center;
}

.waveform {
  width: 100%;
  max-width: 420px;
}

.waveform-bars {
  display: flex;
  align-items: flex-end;
  gap: 3px;
  height: 200px;
  padding: 24px;
  background: var(--surface);
  border: 1px solid var(--hairline);
  border-radius: 2px;
}

.waveform-bar {
  flex: 1;
  min-width: 3px;
  height: var(--bar-h);
  background: var(--slate);
  opacity: 0;
  transform-origin: bottom;
  animation: waveform-reveal 0.4s cubic-bezier(0.23, 1, 0.32, 1) forwards;
  animation-delay: calc(var(--bar-i) * 12ms);
}

.waveform-bar-signal {
  background: var(--signal);
}

@keyframes waveform-reveal {
  from {
    opacity: 0;
    transform: scaleY(0);
  }
  to {
    opacity: 0.85;
    transform: scaleY(1);
  }
}

.waveform-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 12px;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  letter-spacing: 1px;
  text-transform: uppercase;
}

.waveform-label-noise {
  color: var(--slate);
}

.waveform-label-signal {
  color: var(--signal);
}

/* ════════════════════════════════════════════════════════════
   Section label (shared)
   ════════════════════════════════════════════════════════════ */
.section-label {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
}

.section-label-text {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--signal);
  font-weight: 600;
}

.section-label-rule {
  flex: 1;
  height: 1px;
  background: var(--hairline);
}

.section-title {
  font-family: var(--font-display);
  font-weight: 400;
  font-size: clamp(1.8rem, 3vw, 2.4rem);
  line-height: 1.2;
  letter-spacing: -0.5px;
  margin-bottom: 12px;
  color: var(--ink-deep);
}

.section-desc {
  font-size: 1rem;
  color: var(--slate);
  max-width: 620px;
  margin-bottom: 48px;
}

/* ════════════════════════════════════════════════════════════
   Methodology — vertical phase timeline
   ════════════════════════════════════════════════════════════ */
.methodology {
  padding: 80px 0;
  border-top: 1px solid var(--hairline);
}

.phases {
  display: flex;
  flex-direction: column;
  gap: 0;
  max-width: 720px;
}

.phase {
  display: flex;
  gap: 24px;
  align-items: flex-start;
}

.phase-marker {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex-shrink: 0;
  width: 40px;
}

.phase-num {
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 0.85rem;
  color: var(--signal);
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--signal);
  border-radius: 2px;
  background: var(--paper);
  z-index: 1;
}

.phase-line {
  width: 1px;
  height: 48px;
  background: var(--hairline);
  margin-top: 4px;
}

.phase-body {
  padding-bottom: 48px;
  padding-top: 8px;
}

.phase:last-child .phase-body {
  padding-bottom: 0;
}

.phase-title {
  font-family: var(--font-body);
  font-weight: 600;
  font-size: 1.1rem;
  color: var(--ink-deep);
  margin-bottom: 4px;
}

.phase-desc {
  font-size: 0.9rem;
  color: var(--slate);
  line-height: 1.5;
  max-width: 520px;
}

/* ════════════════════════════════════════════════════════════
   Deploy — upload + textarea + button
   ════════════════════════════════════════════════════════════ */
.deploy {
  padding: 80px 0 60px;
  border-top: 1px solid var(--hairline);
}

.deploy-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 32px;
  margin-bottom: 32px;
}

.deploy-card {
  background: var(--white);
  border: 1px solid var(--hairline);
  border-radius: 2px;
  display: flex;
  flex-direction: column;
}

.deploy-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid var(--hairline);
  font-family: var(--font-mono);
  font-size: 0.72rem;
}

.deploy-card-label {
  color: var(--ink-deep);
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.deploy-card-meta {
  color: var(--slate);
  letter-spacing: 0.5px;
}

/* Upload zone */
.upload-zone {
  flex: 1;
  min-height: 180px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background 0.2s, border-color 0.2s;
  border-bottom: 1px solid transparent;
}

.upload-zone--drag {
  background: rgba(196, 117, 26, 0.05);
}

.upload-zone--filled {
  align-items: flex-start;
  cursor: default;
}

.upload-input {
  position: absolute;
  width: 0;
  height: 0;
  opacity: 0;
  pointer-events: none;
}

.upload-empty {
  text-align: center;
  padding: 32px 20px;
}

.upload-empty-icon {
  color: var(--slate);
  margin-bottom: 16px;
  display: flex;
  justify-content: center;
}

.upload-empty-text {
  font-size: 0.9rem;
  color: var(--ink-mid);
  font-weight: 500;
  margin-bottom: 6px;
}

.upload-empty-hint {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--slate);
  letter-spacing: 0.3px;
}

.file-list {
  width: 100%;
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.file-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  background: var(--surface);
  border: 1px solid var(--hairline);
  border-radius: 2px;
  font-family: var(--font-mono);
  font-size: 0.82rem;
  color: var(--ink-deep);
}

.file-item-icon {
  color: var(--signal);
  font-size: 0.7rem;
}

.file-item-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-item-remove {
  background: none;
  border: none;
  cursor: pointer;
  color: var(--slate);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2px;
  transition: color 0.2s;
}

.file-item-remove:hover {
  color: var(--signal);
}

.upload-count {
  padding: 12px 20px;
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--slate);
  border-top: 1px solid var(--hairline);
}

/* Textarea */
.textarea-wrapper {
  flex: 1;
  padding: 20px;
}

.textarea {
  width: 100%;
  border: none;
  background: transparent;
  font-family: var(--font-body);
  font-size: 0.92rem;
  line-height: 1.6;
  resize: none;
  outline: none;
  min-height: 140px;
  color: var(--ink-deep);
}

.textarea::placeholder {
  color: var(--slate);
  opacity: 0.7;
}

/* Deploy button */
.deploy-btn {
  width: 100%;
  background: var(--ink-deep);
  color: var(--white);
  border: 1px solid var(--ink-deep);
  padding: 18px 32px;
  font-family: var(--font-body);
  font-weight: 500;
  font-size: 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  transition: background 0.2s, border-color 0.2s, transform 0.15s;
  border-radius: 2px;
  letter-spacing: 0.3px;
}

.deploy-btn:not(:disabled) {
  animation: deploy-pulse 3s ease-in-out infinite;
}

.deploy-btn:hover:not(:disabled) {
  background: var(--ink-mid);
  border-color: var(--ink-mid);
}

.deploy-btn:active:not(:disabled) {
  transform: translateY(1px);
}

.deploy-btn:disabled {
  background: var(--hairline);
  color: var(--slate);
  cursor: not-allowed;
  border-color: var(--hairline);
  animation: none;
}

.deploy-btn-arrow {
  font-size: 1.2rem;
  line-height: 1;
}

@keyframes deploy-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(10, 27, 42, 0.15); }
  50% { box-shadow: 0 0 0 6px rgba(10, 27, 42, 0); }
}

/* ════════════════════════════════════════════════════════════
   Responsive
   ════════════════════════════════════════════════════════════ */
@media (max-width: 900px) {
  .hero {
    grid-template-columns: 1fr;
    gap: 48px;
    padding: 48px 0 64px;
  }

  .hero-right {
    order: -1;
    justify-content: flex-start;
  }

  .waveform {
    max-width: 100%;
  }

  .deploy-grid {
    grid-template-columns: 1fr;
    gap: 24px;
  }
}

@media (max-width: 640px) {
  .nav-inner {
    padding: 0 20px;
  }

  .content {
    padding: 0 20px;
  }

  .hero {
    padding: 32px 0 48px;
  }

  .hero-cta-row {
    flex-direction: column;
  }

  .cta-primary,
  .cta-secondary {
    justify-content: center;
  }

  .methodology,
  .deploy {
    padding: 48px 0 40px;
  }

  .waveform-bars {
    height: 140px;
    padding: 16px;
  }

  .phase {
    gap: 16px;
  }

  .phase-marker {
    width: 32px;
  }

  .phase-num {
    width: 32px;
    height: 32px;
    font-size: 0.75rem;
  }

  .phase-body {
    padding-top: 4px;
  }
}
</style>
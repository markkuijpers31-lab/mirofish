<template>
  <div class="run-preflight" :class="'state-' + state">
    <div class="pf-head">
      <span class="pf-title">{{ $t('runContract.title') }}</span>
      <span class="pf-badge" :class="'badge-' + state">{{ $t('runContract.readiness.' + state) }}</span>
      <button v-if="refreshable" class="pf-refresh" :disabled="loading" @click="$emit('refresh')">
        {{ loading ? $t('runContract.checking') : $t('runContract.recheck') }}
      </button>
    </div>

    <p v-if="error" class="pf-error">{{ $t('runContract.unavailable', { error }) }}</p>

    <template v-if="preflight">
      <p class="pf-locked">{{ $t('runContract.lockedNote') }}</p>
      <ul class="pf-rows">
        <li v-for="row in rows" :key="row.key + (row.params.label || '')" :class="{ bad: !row.ok }">
          <span class="pf-key">{{ $t('runContract.rows.' + row.key, row.params) }}</span>
          <span class="pf-val">{{ row.value }}</span>
          <span class="pf-mark">{{ row.ok ? '✓' : '✗' }}</span>
        </li>
      </ul>

      <ul v-if="preflight.errors?.length" class="pf-errors">
        <li v-for="(e, i) in preflight.errors" :key="i">{{ e }}</li>
      </ul>

      <details v-if="!compact" class="pf-section" open>
        <summary>{{ $t('runContract.personasTitle', { count: personas.length }) }}</summary>
        <table class="pf-table">
          <thead>
            <tr>
              <th>{{ $t('runContract.colPersona') }}</th>
              <th>{{ $t('runContract.colGraphName') }}</th>
              <th>{{ $t('agentSelection.simulationRole') }}</th>
              <th>{{ $t('agentSelection.evidenceRole') }}</th>
              <th>{{ $t('runContract.colStatus') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="p in personas" :key="p.personaId" :class="{ bad: !p.ok }">
              <td class="mono">{{ p.personaId }}</td>
              <td>{{ p.name }}</td>
              <td class="mono">{{ p.simulationRole }}</td>
              <td class="mono">{{ p.evidenceRole }}</td>
              <td>{{ p.ok ? '✓' : p.status === 'resolved' ? $t('runContract.notInSelection') : p.status }}</td>
            </tr>
          </tbody>
        </table>
      </details>

      <details v-if="!compact" class="pf-section">
        <summary>{{ $t('runContract.stagesTitle', { count: stages.length }) }}</summary>
        <table class="pf-table">
          <thead>
            <tr>
              <th>#</th><th>{{ $t('runContract.colStage') }}</th><th>{{ $t('agentSelection.triggerRound') }}</th>
              <th>{{ $t('agentSelection.targets') }}</th><th>{{ $t('agentSelection.condition') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="s in stages" :key="s.id">
              <td class="mono">{{ s.order }}</td>
              <td>{{ s.name }} <span class="mono muted">{{ s.id }}</span></td>
              <td class="mono">{{ s.trigger_round }}</td>
              <td class="mono">{{ [...s.persona_ids, ...s.simulation_roles, ...s.evidence_roles].join(', ') || '*' }}</td>
              <td class="mono">{{ s.condition }} → {{ s.completion }}</td>
            </tr>
          </tbody>
        </table>
        <p v-if="preflight.experiment_plan?.seed_blocked_terms?.length" class="pf-note">
          {{ $t('runContract.seedBlocked', { terms: preflight.experiment_plan.seed_blocked_terms.join(', ') }) }}
        </p>
      </details>

      <details v-if="!compact" class="pf-section" :open="coverage.some(c => !c.ok)">
        <summary>{{ $t('runContract.coverageTitle') }}</summary>
        <table class="pf-table">
          <thead>
            <tr>
              <th>{{ $t('runContract.colDocument') }}</th><th>{{ $t('runContract.colSection') }}</th>
              <th>{{ $t('runContract.colStatus') }}</th><th>{{ $t('runContract.colMissingTerms') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="c in coverage" :key="c.document + c.section" :class="{ bad: !c.ok }">
              <td>{{ c.document }} <span class="muted">({{ c.files }})</span></td>
              <td class="mono">#{{ c.section }}</td>
              <td>{{ c.status }}</td>
              <td>{{ c.missingTerms || '—' }}</td>
            </tr>
          </tbody>
        </table>
      </details>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { coverageRows, personaRows, preflightSummaryRows, readiness } from '../utils/runContract'

const props = defineProps({
  preflight: Object, // GET /preflight data (contract mode) or null
  error: String, // fetch error: always NOT READY
  loading: Boolean,
  compact: Boolean,
  refreshable: { type: Boolean, default: true },
})

defineEmits(['refresh'])

const state = computed(() => readiness(props.preflight, props.error))
const rows = computed(() => preflightSummaryRows(props.preflight))
const personas = computed(() => personaRows(props.preflight))
const coverage = computed(() => coverageRows(props.preflight))
const stages = computed(() => props.preflight?.experiment_plan?.stages || [])
</script>

<style scoped>
.run-preflight {
  border: 2px solid #000;
  padding: 14px;
  font-family: 'JetBrains Mono', 'Space Grotesk', monospace;
  font-size: 12px;
  background: #FFF;
}
.run-preflight.state-not_ready { border-color: #C62828; }
.run-preflight.state-ready_to_start { border-color: #2E7D32; }
.pf-head { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.pf-title { font-weight: 700; letter-spacing: 1px; }
.pf-badge { padding: 2px 8px; font-weight: 700; }
.badge-not_ready { background: #C62828; color: #FFF; }
.badge-ready_to_prepare { background: #FF5722; color: #FFF; }
.badge-ready_to_start { background: #2E7D32; color: #FFF; }
.pf-refresh { margin-left: auto; border: 1px solid #000; background: #FFF; cursor: pointer; font-family: inherit; font-size: 11px; padding: 2px 8px; }
.pf-error, .pf-errors li, .bad { color: #C62828; }
.pf-error { font-weight: 700; }
.pf-locked { color: #555; margin: 4px 0 8px; }
.pf-rows { list-style: none; padding: 0; margin: 0 0 8px; }
.pf-rows li { display: flex; gap: 8px; padding: 2px 0; border-bottom: 1px dashed #EEE; }
.pf-key { flex: 1; }
.pf-val { font-weight: 700; }
.pf-mark { width: 16px; text-align: right; }
.pf-errors { margin: 6px 0; padding-left: 18px; }
.pf-section { margin-top: 8px; }
.pf-section summary { cursor: pointer; font-weight: 700; }
.pf-table { width: 100%; border-collapse: collapse; margin-top: 6px; }
.pf-table th, .pf-table td { text-align: left; padding: 3px 6px; border-bottom: 1px solid #EEE; vertical-align: top; }
.pf-table tr.bad td { background: #FFEBEE; }
.mono { font-family: 'JetBrains Mono', monospace; }
.muted { color: #888; }
.pf-note { color: #555; margin-top: 4px; }
</style>

<template>
  <div class="agent-selection">
    <!-- Population summary: visible before anything is prepared or paid for -->
    <div class="population-banner" :class="{ warn: warnings.length > 0 }">
      <div class="pop-main">
        <span class="pop-num">{{ summary.active }}</span>
        <span class="pop-label">{{ $t('agentSelection.activeAgents') }}</span>
        <span class="pop-sep">/</span>
        <span class="pop-num muted">{{ summary.total }}</span>
        <span class="pop-label">{{ $t('agentSelection.graphEntities') }}</span>
      </div>
      <div class="pop-sub">
        {{ $t('agentSelection.contextOnlySummary', { count: summary.context }) }}
      </div>
      <div class="chip-row" v-if="summary.bySimulationRole.length">
        <span class="chip-title">{{ $t('agentSelection.simulationRole') }}</span>
        <span v-for="r in summary.bySimulationRole" :key="r.id" class="chip">{{ r.label }} · {{ r.count }}</span>
      </div>
      <div class="chip-row">
        <span class="chip-title">{{ $t('agentSelection.evidenceRole') }}</span>
        <span
          v-for="r in summary.byEvidenceRole"
          :key="r.id"
          class="chip"
          :class="'ev-' + r.id"
        >{{ r.label }} · {{ r.count }}</span>
      </div>
      <ul v-if="warnings.length" class="warnings">
        <li v-for="w in warnings" :key="w.key">⚠ {{ $t('agentSelection.warnings.' + w.key, w.params) }}</li>
      </ul>
      <p v-if="data?.missing_saved_entities?.length" class="warn-line">
        ⚠ {{ $t('agentSelection.missingEntities', { names: data.missing_saved_entities.map(m => m.name).join(', ') }) }}
      </p>
    </div>

    <p v-if="locked && !editing" class="locked-note">
      {{ legacyPrepared ? $t('agentSelection.legacyPrepared', { count: data?.population?.legacy_agents_count || 0 }) : $t('agentSelection.lockedNote') }}
    </p>

    <!-- Controls -->
    <div class="controls" v-if="editable">
      <div class="control-row">
        <input v-model="query" class="input search" type="search" :placeholder="$t('agentSelection.searchPlaceholder')" />
        <select v-model="typeFilter" class="input">
          <option value="">{{ $t('agentSelection.allTypes') }}</option>
          <option v-for="(c, type) in summary.byType" :key="type" :value="type">{{ type }} ({{ c.active }}/{{ c.total }})</option>
        </select>
        <select v-model="viewFilter" class="input">
          <option value="all">{{ $t('agentSelection.viewAll') }}</option>
          <option value="active">{{ $t('agentSelection.viewActive') }}</option>
          <option value="context">{{ $t('agentSelection.viewContext') }}</option>
        </select>
      </div>
      <div class="control-row">
        <button class="mini-btn" @click="setAll(true)">{{ $t('agentSelection.selectAll') }}</button>
        <button class="mini-btn" @click="setAll(false)">{{ $t('agentSelection.selectNone') }}</button>
        <button class="mini-btn" :disabled="!filtered.length" @click="setFiltered({ active: true })">
          {{ $t('agentSelection.activateFiltered', { count: filtered.length }) }}
        </button>
        <button class="mini-btn" :disabled="!filtered.length" @click="setFiltered({ active: false })">
          {{ $t('agentSelection.deactivateFiltered', { count: filtered.length }) }}
        </button>
        <button v-if="briefRows.length" class="mini-btn accent" @click="activateBriefPersonas">
          {{ $t('agentSelection.activateBriefPersonas', { count: briefRows.length }) }}
        </button>
      </div>
      <div class="control-row">
        <span class="control-label">{{ $t('agentSelection.bulkFor', { count: filtered.length }) }}</span>
        <select class="input" :value="''" @change="e => { if (e.target.value) setFiltered({ simulation_role: e.target.value }); e.target.value = '' }">
          <option value="">{{ $t('agentSelection.setSimulationRole') }}</option>
          <option v-for="r in roleCatalog" :key="r.id" :value="r.id">{{ r.label }}</option>
        </select>
        <select class="input" :value="''" @change="e => { if (e.target.value) setFiltered({ evidence_role: e.target.value }); e.target.value = '' }">
          <option value="">{{ $t('agentSelection.setEvidenceRole') }}</option>
          <option v-for="r in evidenceRoles" :key="r.id" :value="r.id">{{ r.label }}</option>
        </select>
      </div>
    </div>

    <!-- ACTIVE AGENTS -->
    <div class="entity-section active-section">
      <div class="section-head">
        <span class="section-title">{{ $t('agentSelection.activeSection') }}</span>
        <span class="section-count">{{ activeRows.length }}<template v-if="filtersOn"> / {{ summary.active }}</template></span>
      </div>
      <p class="section-desc">{{ $t('agentSelection.activeSectionDesc') }}</p>
      <div v-if="!activeRows.length" class="empty">{{ $t('agentSelection.noActiveRows') }}</div>
      <table v-else class="entity-table">
        <thead>
          <tr>
            <th class="col-check">{{ $t('agentSelection.colActive') }}</th>
            <th>{{ $t('agentSelection.colName') }}</th>
            <th>{{ $t('agentSelection.colType') }}</th>
            <th>{{ $t('agentSelection.colDescription') }}</th>
            <th>{{ $t('agentSelection.simulationRole') }}</th>
            <th>{{ $t('agentSelection.evidenceRole') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="e in activeRows" :key="e.entity_uuid" class="row-active">
            <td class="col-check"><input type="checkbox" :checked="e.active" :disabled="!editable" @change="toggle(e)" /></td>
            <td class="col-name">
              <span class="entity-name">{{ e.name }}</span>
              <span v-if="e.brief_persona_id" class="tag brief">{{ $t('agentSelection.inBrief', { id: e.brief_persona_id }) }}</span>
              <span v-if="e.duplicate_of" class="tag dup" :title="e.duplicate_reason">{{ $t('agentSelection.aliasOf', { name: e.duplicate_of_name }) }}</span>
              <span v-for="p in e.possible_duplicates" :key="p.uuid" class="tag maybe" :title="p.reason">{{ $t('agentSelection.possibleDuplicate', { name: p.name }) }}</span>
            </td>
            <td class="col-type">{{ e.entity_type }}</td>
            <td class="col-desc">{{ e.description || '—' }}</td>
            <td>
              <select class="input small" :value="e.simulation_role" :disabled="!editable" @change="setRow(e, { simulation_role: $event.target.value })">
                <option v-for="r in roleCatalog" :key="r.id" :value="r.id">{{ r.label }}</option>
              </select>
            </td>
            <td>
              <select class="input small" :class="'ev-' + e.evidence_role" :value="e.evidence_role" :disabled="!editable" @change="setRow(e, { evidence_role: $event.target.value })">
                <option v-for="r in evidenceRoles" :key="r.id" :value="r.id">{{ r.label }}</option>
              </select>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- CONTEXT-ONLY ENTITIES -->
    <div class="entity-section context-section">
      <div class="section-head">
        <span class="section-title">{{ $t('agentSelection.contextSection') }}</span>
        <span class="section-count">{{ contextRows.length }}<template v-if="filtersOn"> / {{ summary.context }}</template></span>
      </div>
      <p class="section-desc">{{ $t('agentSelection.contextSectionDesc') }}</p>
      <div v-if="!contextRows.length" class="empty">{{ $t('agentSelection.noContextRows') }}</div>
      <table v-else class="entity-table">
        <thead>
          <tr>
            <th class="col-check">{{ $t('agentSelection.colActive') }}</th>
            <th>{{ $t('agentSelection.colName') }}</th>
            <th>{{ $t('agentSelection.colType') }}</th>
            <th>{{ $t('agentSelection.colDescription') }}</th>
            <th>{{ $t('agentSelection.suggestedRole') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="e in contextRows" :key="e.entity_uuid">
            <td class="col-check"><input type="checkbox" :checked="e.active" :disabled="!editable" @change="toggle(e)" /></td>
            <td class="col-name">
              <span class="entity-name">{{ e.name }}</span>
              <span v-if="e.brief_persona_id" class="tag brief">{{ $t('agentSelection.inBrief', { id: e.brief_persona_id }) }}</span>
              <span v-if="e.duplicate_of" class="tag dup" :title="e.duplicate_reason">{{ $t('agentSelection.aliasOf', { name: e.duplicate_of_name }) }}</span>
              <span v-for="p in e.possible_duplicates" :key="p.uuid" class="tag maybe" :title="p.reason">{{ $t('agentSelection.possibleDuplicate', { name: p.name }) }}</span>
            </td>
            <td class="col-type">{{ e.entity_type }}</td>
            <td class="col-desc">{{ e.description || '—' }}</td>
            <td>
              <select class="input small" :value="e.simulation_role" :disabled="!editable" @change="setRow(e, { simulation_role: $event.target.value })">
                <option v-for="r in roleCatalog" :key="r.id" :value="r.id">{{ r.label }}</option>
              </select>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Simulation role catalog (configuration, editable per simulation) -->
    <details class="sub-panel" v-if="editable">
      <summary>{{ $t('agentSelection.rolesTitle', { count: roleCatalog.length }) }}</summary>
      <p class="section-desc">{{ $t('agentSelection.rolesDesc') }}</p>
      <div class="role-list">
        <span v-for="r in roleCatalog" :key="r.id" class="chip">
          {{ r.label }} → {{ evidenceLabel(r.default_evidence_role) }}
          <button v-if="!roleInUse(r.id)" class="chip-x" :title="$t('agentSelection.removeRole')" @click="removeRole(r.id)">×</button>
        </span>
      </div>
      <div class="control-row">
        <input v-model="newRoleLabel" class="input" :placeholder="$t('agentSelection.newRolePlaceholder')" />
        <select v-model="newRoleEvidence" class="input">
          <option v-for="r in evidenceRoles" :key="r.id" :value="r.id">{{ r.label }}</option>
        </select>
        <button class="mini-btn" :disabled="!newRoleLabel.trim()" @click="addRole">{{ $t('agentSelection.addRole') }}</button>
      </div>
    </details>

    <!-- Private experiment stages -->
    <details class="sub-panel" :open="stages.length > 0">
      <summary>
        {{ $t('agentSelection.stagesTitle', { count: stages.length }) }}
        <span v-if="!planSaved && stages.length" class="tag brief">{{ $t('agentSelection.fromBrief') }}</span>
      </summary>
      <p class="section-desc">{{ $t('agentSelection.stagesDesc') }}</p>
      <div v-for="(s, idx) in stages" :key="idx" class="stage-card">
        <div class="stage-row">
          <input v-model="s.name" class="input grow" :placeholder="$t('agentSelection.stageName')" :disabled="!planEditable" />
          <label class="inline-label">{{ $t('agentSelection.triggerRound') }}
            <input v-model.number="s.trigger_round" type="number" min="1" class="input num" :disabled="!planEditable" />
          </label>
          <span v-if="s.source === 'brief'" class="tag brief">{{ $t('agentSelection.fromBrief') }}</span>
          <button v-if="planEditable" class="mini-btn danger" @click="stages.splice(idx, 1)">{{ $t('agentSelection.removeStage') }}</button>
        </div>
        <div class="stage-row">
          <span class="control-label">{{ $t('agentSelection.targets') }}</span>
          <label v-for="r in roleCatalog" :key="r.id" class="check-label">
            <input type="checkbox" :value="r.id" v-model="s.target.simulation_roles" :disabled="!planEditable" /> {{ r.label }}
          </label>
          <input v-model="s.personaIdsText" class="input" :placeholder="$t('agentSelection.personaIds')" :disabled="!planEditable" />
        </div>
        <textarea v-model="s.private_instruction" class="input textarea" rows="3" :placeholder="$t('agentSelection.privateInstruction')" :disabled="!planEditable"></textarea>
        <p v-if="s.context_sections?.length || s.context || s.records?.length" class="stage-meta">
          <span v-if="s.context_sections?.length">{{ $t('agentSelection.contextSections', { ids: s.context_sections.join(', ') }) }}</span>
          <span v-if="s.context">{{ $t('agentSelection.contextChars', { count: s.context.length }) }}</span>
          <span v-if="s.records?.length">{{ $t('agentSelection.records', { labels: s.records.join(', ') }) }}</span>
        </p>
        <div class="stage-row">
          <span class="control-label">{{ $t('agentSelection.condition') }}</span>
          <select v-model="s.condition.type" class="input" :disabled="!planEditable">
            <option v-for="c in conditionTypes" :key="c" :value="c">{{ $t('agentSelection.conditionTypes.' + c) }}</option>
          </select>
          <input v-if="['mentioned', 'not_mentioned', 'authored_mentioned', 'authored_not_mentioned'].includes(s.condition.type)" v-model="s.condition.termsText" class="input" :placeholder="$t('agentSelection.terms')" :disabled="!planEditable" />
          <select v-if="['stage_completed', 'stage_not_completed'].includes(s.condition.type)" v-model="s.condition.stage_id" class="input" :disabled="!planEditable">
            <option v-for="o in stages.filter(x => x !== s)" :key="o.id" :value="o.id">{{ o.name || o.id }}</option>
          </select>
          <span class="control-label">{{ $t('agentSelection.completion') }}</span>
          <select v-model="s.completion.type" class="input" :disabled="!planEditable">
            <option v-for="c in completionTypes" :key="c" :value="c">{{ $t('agentSelection.completionTypes.' + c) }}</option>
          </select>
          <input v-if="['mentioned', 'authored_mentioned'].includes(s.completion.type)" v-model="s.completion.termsText" class="input" :placeholder="$t('agentSelection.terms')" :disabled="!planEditable" />
          <input v-if="s.completion.type === 'action_types'" v-model="s.completion.actionTypesText" class="input" placeholder="CREATE_POST, CREATE_COMMENT" :disabled="!planEditable" />
          <input v-if="s.completion.type === 'rounds_elapsed'" v-model.number="s.completion.rounds" type="number" min="1" class="input num" :disabled="!planEditable" />
        </div>
      </div>
      <div class="control-row" v-if="planEditable">
        <button class="mini-btn" @click="addStage">{{ $t('agentSelection.addStage') }}</button>
        <button class="mini-btn accent" :disabled="savingPlan" @click="savePlan">{{ $t('agentSelection.saveStages') }}</button>
        <span v-if="planMessage" class="plan-message" :class="{ error: planError }">{{ planMessage }}</span>
      </div>
    </details>

    <!-- Confirm -->
    <div class="confirm-row">
      <template v-if="editable">
        <button class="action-btn primary" :disabled="summary.active === 0 || saving || busy" @click="confirm">
          {{ $t('agentSelection.confirmAndPrepare', { active: summary.active, total: summary.total }) }}
        </button>
        <button v-if="locked" class="action-btn secondary" :disabled="saving" @click="cancelEdit">{{ $t('agentSelection.cancelEdit') }}</button>
        <span class="confirm-note">{{ $t('agentSelection.confirmNote') }}</span>
      </template>
      <template v-else>
        <button class="action-btn secondary" :disabled="busy" @click="startEdit">{{ $t('agentSelection.changePopulation') }}</button>
        <span class="confirm-note">{{ $t('agentSelection.changeNote') }}</span>
      </template>
      <p v-if="error" class="error-line">{{ error }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { saveAgentSelection, getExperimentPlan, saveExperimentPlan } from '../api/simulation'
import {
  applyToEntities,
  filterEntities,
  populationWarnings,
  selectionPayload,
  summarizeSelection,
} from '../utils/agentSelection'

const { t } = useI18n()

const props = defineProps({
  simulationId: String,
  selection: Object, // GET /agent-selection response
  locked: Boolean, // population already prepared: read-only until "change"
  busy: Boolean, // preparation running
})

const emit = defineEmits(['confirmed', 'add-log'])

const data = ref(null)
const rows = ref([])
const roleCatalog = ref([])
const evidenceRoles = ref([])
const query = ref('')
const typeFilter = ref('')
const viewFilter = ref('all')
const editing = ref(false)
const saving = ref(false)
const error = ref('')
const newRoleLabel = ref('')
const newRoleEvidence = ref('influencer_context')

const stages = ref([])
const planSaved = ref(false)
const conditionTypes = ref(['always'])
const completionTypes = ref(['on_delivery'])
const savingPlan = ref(false)
const planMessage = ref('')
const planError = ref(false)

const editable = computed(() => (!props.locked || editing.value) && !props.busy)
const planEditable = computed(() => !props.busy)
const legacyPrepared = computed(() => Boolean(data.value?.population?.legacy_prepared))

const loadFrom = (selection) => {
  data.value = selection
  rows.value = (selection?.entities || []).map(e => ({ ...e }))
  roleCatalog.value = (selection?.role_catalog || []).map(r => ({ ...r }))
  evidenceRoles.value = selection?.evidence_roles || []
}

watch(() => props.selection, (s) => {
  if (s) loadFrom(s)
}, { immediate: true })

const summary = computed(() => summarizeSelection(rows.value, roleCatalog.value, evidenceRoles.value))
const warnings = computed(() => populationWarnings(rows.value))
const filtersOn = computed(() => Boolean(query.value || typeFilter.value || viewFilter.value !== 'all'))
const filtered = computed(() => filterEntities(rows.value, { query: query.value, type: typeFilter.value, view: viewFilter.value }))
const activeRows = computed(() => filtered.value.filter(e => e.active))
const contextRows = computed(() => filtered.value.filter(e => !e.active))
const briefRows = computed(() => rows.value.filter(e => e.brief_persona_id))

const evidenceLabel = (id) => evidenceRoles.value.find(r => r.id === id)?.label || id
const roleInUse = (id) => rows.value.some(e => e.simulation_role === id)

const update = (uuids, patch) => {
  rows.value = applyToEntities(rows.value, uuids, patch, roleCatalog.value)
}
const toggle = (e) => update([e.entity_uuid], { active: !e.active })
const setRow = (e, patch) => update([e.entity_uuid], patch)
const setAll = (active) => update(rows.value.map(e => e.entity_uuid), { active })
const setFiltered = (patch) => update(filtered.value.map(e => e.entity_uuid), patch)
const activateBriefPersonas = () => update(briefRows.value.map(e => e.entity_uuid), { active: true })

const addRole = () => {
  const label = newRoleLabel.value.trim()
  const id = label.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40)
  if (!id || roleCatalog.value.some(r => r.id === id)) return
  roleCatalog.value = [...roleCatalog.value, { id, label, default_evidence_role: newRoleEvidence.value }]
  newRoleLabel.value = ''
}
const removeRole = (id) => {
  roleCatalog.value = roleCatalog.value.filter(r => r.id !== id)
}

const errorText = (err) => err?.response?.data?.error || err?.message || t('common.unknownError')

const startEdit = () => {
  editing.value = true
}
const cancelEdit = () => {
  editing.value = false
  loadFrom(props.selection)
}

const confirm = async () => {
  if (!props.simulationId || summary.value.active === 0) return
  saving.value = true
  error.value = ''
  try {
    const res = await saveAgentSelection(props.simulationId, selectionPayload(rows.value, roleCatalog.value, true))
    loadFrom(res.data)
    editing.value = false
    emit('add-log', t('agentSelection.logConfirmed', { active: res.data.summary.active_agents, total: res.data.summary.graph_entities }))
    emit('confirmed', res.data)
  } catch (err) {
    error.value = errorText(err)
  } finally {
    saving.value = false
  }
}

// ---- experiment stages ----

const toEditable = (s) => ({
  ...s,
  target: { agent_ids: [], persona_ids: [], simulation_roles: [], evidence_roles: [], ...(s.target || {}) },
  personaIdsText: (s.target?.persona_ids || []).join(', '),
  condition: { type: 'always', ...(s.condition || {}), termsText: (s.condition?.terms || []).join(', ') },
  completion: {
    type: 'on_delivery', rounds: 1, ...(s.completion || {}),
    termsText: (s.completion?.terms || []).join(', '),
    actionTypesText: (s.completion?.action_types || []).join(', '),
  },
})

const splitList = (text) => (text || '').split(',').map(x => x.trim()).filter(Boolean)

const loadPlan = async () => {
  if (!props.simulationId) return
  try {
    const res = await getExperimentPlan(props.simulationId)
    stages.value = (res.data.stages || []).map(toEditable)
    planSaved.value = Boolean(res.data.saved)
    conditionTypes.value = res.data.condition_types || conditionTypes.value
    completionTypes.value = res.data.completion_types || completionTypes.value
  } catch (err) {
    planMessage.value = errorText(err)
    planError.value = true
  }
}

const addStage = () => {
  const n = stages.value.length + 1
  stages.value.push(toEditable({ id: `stage_${Date.now().toString(36)}`, name: `Stage ${n}`, order: n, trigger_round: 1, private_instruction: '', source: 'user' }))
}

const savePlan = async () => {
  savingPlan.value = true
  planMessage.value = ''
  planError.value = false
  try {
    const payload = {
      stages: stages.value.map((s, i) => ({
        id: s.id,
        name: s.name,
        order: i + 1,
        trigger_round: s.trigger_round,
        target: { ...s.target, persona_ids: splitList(s.personaIdsText) },
        private_instruction: s.private_instruction,
        // Fields this editor does not edit are passed through unchanged
        context: s.context || '',
        context_sections: s.context_sections || [],
        records: s.records || [],
        condition: {
          type: s.condition.type,
          terms: splitList(s.condition.termsText),
          stage_id: s.condition.type.startsWith('stage_') ? s.condition.stage_id : undefined,
        },
        completion: {
          type: s.completion.type,
          terms: splitList(s.completion.termsText),
          action_types: splitList(s.completion.actionTypesText),
          rounds: s.completion.rounds,
        },
        source: s.source,
      })),
    }
    const res = await saveExperimentPlan(props.simulationId, payload)
    stages.value = (res.data.stages || []).map(toEditable)
    planSaved.value = true
    planMessage.value = t('agentSelection.stagesSaved')
    emit('add-log', t('agentSelection.logStagesSaved', { count: stages.value.length }))
  } catch (err) {
    planMessage.value = errorText(err)
    planError.value = true
  } finally {
    savingPlan.value = false
  }
}

watch(() => props.simulationId, () => loadPlan(), { immediate: true })
</script>

<style scoped>
.agent-selection {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.population-banner {
  border: 1px solid #EAEAEA;
  border-left: 4px solid #000;
  border-radius: 6px;
  padding: 14px 16px;
  background: #FAFAFA;
}
.population-banner.warn {
  border-left-color: #FF5722;
  background: #FFF8F5;
}
.pop-main {
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
}
.pop-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 28px;
  font-weight: 700;
  color: #000;
}
.pop-num.muted { color: #999; }
.pop-label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #333;
}
.pop-sep { font-size: 22px; color: #CCC; }
.pop-sub { font-size: 12px; color: #666; margin: 4px 0 8px; }

.chip-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 6px; }
.chip-title {
  font-size: 10px;
  font-weight: 600;
  color: #999;
  text-transform: uppercase;
  min-width: 110px;
}
.chip {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 10px;
  background: #F0F0F0;
  color: #333;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.chip.ev-target_consumer, .input.ev-target_consumer { background: #E8F5E9; color: #2E7D32; }
.chip.ev-influencer_context, .input.ev-influencer_context { background: #E3F2FD; color: #1565C0; }
.chip.ev-excluded, .input.ev-excluded { background: #F5F5F5; color: #757575; }
.chip-x { border: none; background: transparent; cursor: pointer; color: #999; font-size: 13px; padding: 0 2px; }

.warnings { margin: 10px 0 0; padding: 0; list-style: none; }
.warnings li, .warn-line { font-size: 12px; color: #D84315; font-weight: 600; margin: 4px 0 0; }

.locked-note { font-size: 12px; color: #2E7D32; background: #E8F5E9; padding: 8px 12px; border-radius: 4px; margin: 0; }

.controls { display: flex; flex-direction: column; gap: 8px; }
.control-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.control-label { font-size: 11px; color: #666; font-weight: 600; }
.input {
  font-family: inherit;
  font-size: 12px;
  padding: 6px 8px;
  border: 1px solid #DDD;
  border-radius: 4px;
  background: #FFF;
  color: #333;
}
.input.search { flex: 1; min-width: 180px; }
.input.small { padding: 4px 6px; font-size: 11px; max-width: 170px; }
.input.num { width: 64px; }
.input.grow { flex: 1; min-width: 160px; }
.input.textarea { width: 100%; box-sizing: border-box; resize: vertical; margin: 6px 0; }
.mini-btn {
  font-family: inherit;
  font-size: 11px;
  font-weight: 600;
  padding: 6px 10px;
  border: 1px solid #DDD;
  border-radius: 4px;
  background: #FFF;
  cursor: pointer;
}
.mini-btn:hover:not(:disabled) { border-color: #000; }
.mini-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.mini-btn.accent { background: #000; color: #FFF; border-color: #000; }
.mini-btn.danger { color: #C62828; }

.entity-section { border: 1px solid #EAEAEA; border-radius: 6px; overflow-x: auto; }
.active-section { border-color: #000; }
.section-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 12px;
  background: #F5F5F5;
}
.active-section .section-head { background: #000; color: #FFF; }
.section-title { font-size: 12px; font-weight: 700; letter-spacing: 1px; }
.section-count { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; }
.section-desc { font-size: 11px; color: #777; margin: 8px 12px; line-height: 1.5; }
.sub-panel .section-desc { margin: 8px 0; }
.empty { font-size: 12px; color: #999; padding: 12px; }

.entity-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.entity-table th {
  text-align: left;
  font-size: 10px;
  text-transform: uppercase;
  color: #999;
  font-weight: 600;
  padding: 6px 8px;
  border-bottom: 1px solid #EEE;
}
.entity-table td { padding: 6px 8px; border-bottom: 1px solid #F3F3F3; vertical-align: top; }
.entity-table tr.row-active td { background: #FFFDFB; }
.col-check { width: 44px; text-align: center; }
.col-name { min-width: 160px; }
.col-type { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #555; white-space: nowrap; }
.col-desc { color: #666; min-width: 140px; max-width: 260px; line-height: 1.4; }
.entity-name { font-weight: 600; color: #000; margin-right: 6px; }
.tag {
  display: inline-block;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 3px;
  margin: 2px 4px 0 0;
  font-weight: 600;
}
.tag.brief { background: #FFF3E0; color: #E65100; }
.tag.dup { background: #FCE4EC; color: #AD1457; }
.tag.maybe { background: #FFFDE7; color: #8D6E00; }

.sub-panel { border: 1px solid #EAEAEA; border-radius: 6px; padding: 10px 12px; }
.sub-panel summary { cursor: pointer; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; }
.role-list { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.stage-card { border: 1px dashed #DDD; border-radius: 6px; padding: 10px; margin-bottom: 10px; }
.stage-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 6px; }
.inline-label, .check-label { font-size: 11px; color: #555; display: inline-flex; align-items: center; gap: 4px; }
.plan-message { font-size: 11px; color: #2E7D32; }
.stage-meta { font-size: 11px; color: #555; margin: 0 0 6px; display: flex; flex-wrap: wrap; gap: 10px; }
.plan-message.error { color: #C62828; }

.confirm-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
.confirm-note { font-size: 11px; color: #777; flex: 1; min-width: 200px; }
.action-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  font-size: 13px;
  font-weight: 600;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}
.action-btn.primary { background: #000; color: #FFF; }
.action-btn.primary:hover:not(:disabled) { background: #FF5722; }
.action-btn.secondary { background: #F5F5F5; color: #333; }
.action-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.error-line { width: 100%; font-size: 12px; color: #C62828; margin: 0; }
</style>

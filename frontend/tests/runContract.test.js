// Run Contract / preflight UI regression tests (run with: npm test)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  READINESS,
  afterSelectionSaved,
  canContinueToRun,
  canPrepare,
  canStartRun,
  coverageRows,
  isContractMode,
  lockedRounds,
  personaRows,
  preflightFetchState,
  preflightSummaryRows,
  readiness,
  runRouteQuery,
  selectionEditable,
  startRequest,
} from '../src/utils/runContract.js'

const SRC = join(dirname(fileURLToPath(import.meta.url)), '..', 'src')
const read = (rel) => readFileSync(join(SRC, rel), 'utf8')

const IDS = [
  ...Array.from({ length: 16 }, (_, i) => `C${String(i + 1).padStart(2, '0')}`),
  'U01', 'U02', 'U03', 'U04', 'U05', 'A01', 'A02', 'A03',
]

// Shape of GET /preflight for the Kamptribe contract (as the backend returns it)
const preflightFixture = ({ prepared = false } = {}) => ({
  mode: 'contract',
  contract: { present: true, valid: true, applied: true, id: 'kamptribe-run-2c-final', hash: 'h' },
  population: {
    expected: 24, resolved: 24, active: 24, ok: true,
    missing_persona_ids: [], duplicate_persona_ids: [], unexpected_active: [],
    role_mismatches: [], evidence_role_mismatches: [],
    entities: IDS.map(id => ({
      persona_id: id, status: 'resolved', name: `${id} - persona`,
      simulation_role: id.startsWith('C') ? 'prospect_cold' : 'brand_advocate',
      evidence_role: id.startsWith('C') ? 'target_consumer' : 'influencer_context',
      selection_active: true,
      selection_simulation_role: id.startsWith('C') ? 'prospect_cold' : 'brand_advocate',
      selection_evidence_role: id.startsWith('C') ? 'target_consumer' : 'influencer_context',
    })),
  },
  roles: {
    target_consumers: 16, influencer_context: 8, excluded_active: 0,
    expected_target_consumers: 16, expected_influencer_context: 8, expected_excluded_active: 0, ok: true,
  },
  experiment_plan: { expected_stages: 7, saved_stages: 7, mismatches: [], ok: true, stages: [] },
  rounds: { expected: 50, ok: true },
  sources: {
    ok: true, coverage_ok: true, missing: [],
    documents: [{ label: 'Product Truth', ok: true, files: ['kamptribe_product_truth.md'], sections: [
      { id: 'pricing', status: 'covered', missing_terms: [] },
      { id: 'trip-pass', status: 'covered', missing_terms: [] },
    ] }],
  },
  preparation: prepared
    ? { prepared: true, ok: true, expected_agents: 24, roster_active: 24, profile_count_twitter: 24,
      profile_count_reddit: 24, config_agent_count: 24, profile_persona_ids_match: true,
      config_agents_match: true, roster_matches_contract: true, selection_hash_matches: true,
      roster_contract_hash_matches: true }
    : { prepared: false, ok: false, expected_agents: 24, roster_active: null },
  ready_to_prepare: true,
  ready_to_start: prepared,
  errors: prepared ? [] : ['Not prepared yet for this contract (Prepare is required before Start)'],
})

const row = (rows, key, label) => rows.find(r => r.key === key && (!label || r.params.label === label))

test('the preflight summary shows the exact contract counts before preparation', () => {
  const rows = preflightSummaryRows(preflightFixture())
  assert.equal(row(rows, 'contract').value, 'kamptribe-run-2c-final')
  assert.equal(row(rows, 'activeRequested').value, '24 / 24')
  assert.equal(row(rows, 'targetConsumers').value, '16')
  assert.equal(row(rows, 'influencerContext').value, '8')
  assert.equal(row(rows, 'excludedActive').value, '0')
  assert.equal(row(rows, 'unexpectedActive').value, 'none')
  assert.equal(row(rows, 'stages').value, '7 / 7')
  assert.equal(row(rows, 'rounds').value, '50')
  assert.equal(row(rows, 'coverage', 'Product Truth').value, 'PASS')
  assert.ok(rows.every(r => r.ok))
  assert.equal(row(rows, 'preparedRoster'), undefined)
  assert.equal(readiness(preflightFixture()), READINESS.READY_TO_PREPARE)
})

test('after preparation the summary shows roster, both profile files, config and MATCH', () => {
  const rows = preflightSummaryRows(preflightFixture({ prepared: true }))
  for (const key of ['preparedRoster', 'twitterProfiles', 'redditProfiles', 'configAgents']) {
    assert.equal(row(rows, key).value, '24 / 24', key)
  }
  assert.equal(row(rows, 'selectionRosterContract').value, 'MATCH')
  assert.equal(readiness(preflightFixture({ prepared: true })), READINESS.READY_TO_START)
})

test('every requested persona is listed with graph name and both roles', () => {
  const rows = personaRows(preflightFixture())
  assert.deepEqual(rows.map(r => r.personaId), IDS)
  assert.ok(rows.every(r => r.ok && r.name.endsWith('persona') && r.simulationRole && r.evidenceRole))
})

test('failures are red with the exact ids and names', () => {
  const pf = preflightFixture()
  pf.population.ok = false
  pf.population.unexpected_active = [{ name: 'Kamptribe' }, { name: 'Roadtrippers' }]
  pf.population.entities[2].selection_simulation_role = 'prospect_switcher'
  pf.population.entities[15] = { persona_id: 'C16', status: 'missing', candidates: [] }
  pf.ready_to_prepare = false
  const rows = preflightSummaryRows(pf)
  assert.equal(row(rows, 'unexpectedActive').value, 'Kamptribe, Roadtrippers')
  assert.equal(row(rows, 'unexpectedActive').ok, false)
  assert.equal(row(rows, 'activeRequested').ok, false)
  const personas = personaRows(pf)
  assert.equal(personas[2].ok, false)
  assert.equal(personas[15].ok, false)
  assert.equal(readiness(pf), READINESS.NOT_READY)
})

test('frontend shows the source coverage matrix and its failures (29)', () => {
  const pf = preflightFixture()
  pf.sources.documents[0].ok = false
  pf.sources.documents[0].sections[0] = { id: 'pricing', status: 'incomplete', missing_terms: ['19.99'] }
  const rows = coverageRows(pf)
  assert.deepEqual(rows.map(r => [r.section, r.status, r.ok]), [['pricing', 'incomplete', false], ['trip-pass', 'covered', true]])
  assert.equal(rows[0].missingTerms, '19.99')
  assert.equal(row(preflightSummaryRows(pf), 'coverage', 'Product Truth').value, 'FAIL')
  pf.sources.documents[0].files = []
  assert.ok(coverageRows(pf).every(r => !r.ok && r.files === 'not uploaded'))
})

test('the contracted population cannot be edited (16)', () => {
  assert.equal(selectionEditable({ contractLocked: true }), false)
  assert.equal(selectionEditable({ contractLocked: true, editing: true }), false)
  assert.equal(selectionEditable({ contractLocked: false }), true)
  assert.equal(selectionEditable({ locked: true }), false)
  assert.equal(selectionEditable({ locked: true, editing: true }), true)
  const panel = readFileSync(join(SRC, 'components', 'AgentSelectionPanel.vue'), 'utf8')
  assert.match(panel, /planEditable = computed\(\(\) => !props\.busy && !props\.contractLocked\)/)
  assert.match(panel, /class="confirm-row" v-if="!contractLocked"/)
  assert.match(panel, /class="controls" v-if="editable"/)
})

test('the contract run length is locked and never transported or sent (34)', () => {
  const pf = preflightFixture({ prepared: true })
  assert.equal(lockedRounds(pf), 50)
  assert.deepEqual(runRouteQuery(pf, 40), {})
  assert.deepEqual(startRequest('sim_1', pf, { maxRounds: 40 }), { simulation_id: 'sim_1' })
  assert.deepEqual(startRequest('sim_1', pf, { maxRounds: 40, newRun: true }), { simulation_id: 'sim_1', new_run: true })
  // simulations without a contract keep the configurable length
  assert.equal(lockedRounds({ mode: 'standard', contract: { present: false } }), null)
  assert.deepEqual(runRouteQuery(null, 40), { maxRounds: 40 })
  assert.deepEqual(startRequest('sim_1', null, { maxRounds: 40 }), { simulation_id: 'sim_1', platform: 'parallel', max_rounds: 40 })
})

test('saving or confirming the population never prepares (35)', () => {
  assert.equal(afterSelectionSaved(), 'reload_state')
  const step2 = read('components/Step2EnvSetup.vue')
  const body = (name) => {
    const start = step2.indexOf(`const ${name} = async`)
    assert.ok(start >= 0, name)
    return step2.slice(start, step2.indexOf('\n}\n', start))
  }
  for (const name of ['handleSelectionConfirmed', 'handleSelectionSaveFailed', 'reloadPersistedState', 'handleApplyContract']) {
    assert.doesNotMatch(body(name), /startPrepareSimulation|prepareSimulation\(/, name)
  }
  // startPrepareSimulation is only reached from the explicit Prepare click
  const calls = step2.match(/startPrepareSimulation\(\)/g) || []
  assert.equal(calls.length, 1)
  assert.match(body('handlePrepareClick'), /await startPrepareSimulation\(\)/)
  // mounting never POSTs /prepare
  const mount = step2.slice(step2.indexOf('onMounted(async'), step2.indexOf('onUnmounted('))
  assert.doesNotMatch(mount, /startPrepareSimulation|prepareSimulation/)
})

test('Prepare and Continue follow the backend preflight only', () => {
  const pf = preflightFixture()
  assert.equal(canPrepare({ preflight: pf }), true)
  assert.equal(canPrepare({ preflight: pf, preparing: true }), false)
  assert.equal(canPrepare({ preflight: { ...pf, ready_to_prepare: false } }), false)
  assert.equal(canPrepare({ preflight: pf, error: 'Network Error' }), false)
  assert.equal(canContinueToRun({ preflight: pf, prepared: true }), false)
  assert.equal(canContinueToRun({ preflight: preflightFixture({ prepared: true }), prepared: true }), true)
  assert.equal(canContinueToRun({ preflight: preflightFixture({ prepared: true }), prepared: false }), false)
  assert.equal(canStartRun({ preflight: pf }), false)
  assert.equal(canStartRun({ preflight: preflightFixture({ prepared: true }) }), true)
  // a contracted simulation whose preflight has not loaded is never ready
  assert.equal(canPrepare({ preflight: { contract: { present: true } } }), false)
  // standard simulations: explicit prepare after a confirmed selection
  assert.equal(canPrepare({ preflight: { mode: 'standard', contract: { present: false } }, selectionConfirmed: true }), true)
  assert.equal(canPrepare({ preflight: { mode: 'standard', contract: { present: false } }, selectionConfirmed: false }), false)
})

test('Step 2 to Step 3 navigation, mount, reload and back/forward create no run (43-46)', () => {
  const files = []
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      const path = join(dir, name)
      if (statSync(path).isDirectory()) walk(path)
      else files.push(path)
    }
  }
  walk(SRC)
  for (const path of files) {
    const text = readFileSync(path, 'utf8')
    assert.doesNotMatch(text, /pendingSimulationStart|requestSimulationStart|consumeSimulationStartRequest/, path)
  }
  const step3 = read('components/Step3Simulation.vue')
  const mount = step3.slice(step3.indexOf('onMounted(async'), step3.indexOf('onUnmounted('))
  assert.doesNotMatch(mount, /doStartSimulation|startSimulation\(/)
  // startSimulation is only called from doStartSimulation, which re-fetches the preflight first
  const start = step3.slice(step3.indexOf('const doStartSimulation = async'), step3.indexOf('const handleStartClick'))
  assert.ok(start.indexOf('await loadPreflight()') < start.indexOf('startSimulation(params)'))
  assert.match(start, /if \(!canStartRun\(\{ preflight: fresh, error: preflightError\.value \}\)\) \{/)
  const view = read('views/SimulationView.vue')
  assert.doesNotMatch(view, /requestSimulationStart/)
})

test('a network error leaves the UI NOT READY and never keeps an earlier green state (67, 68)', () => {
  const failed = preflightFetchState({ error: new Error('Network Error') })
  assert.equal(failed.preflight, null)
  assert.equal(failed.error, 'Network Error')
  assert.equal(readiness(failed.preflight, failed.error), READINESS.NOT_READY)
  assert.equal(canPrepare({ ...failed }), false)
  assert.equal(canStartRun({ ...failed }), false)
  assert.equal(readiness(preflightFixture({ prepared: true }), 'timeout'), READINESS.NOT_READY)
  assert.deepEqual(preflightFetchState({ data: null }).preflight, null)
  // a failed selection save makes Step 2 re-read the persisted selection
  const step2 = read('components/Step2EnvSetup.vue')
  assert.match(step2, /@save-failed="handleSelectionSaveFailed"/)
  const panel = read('components/AgentSelectionPanel.vue')
  assert.match(panel, /emit\('save-failed', error\.value\)/)
  assert.match(panel, /stages\.value = \[\]\n    planSaved\.value = false/)
})

test('after a successful mutation the persisted backend state is re-fetched (69)', () => {
  const step2 = read('components/Step2EnvSetup.vue')
  const reload = step2.slice(step2.indexOf('const reloadPersistedState = async'))
  assert.match(reload.slice(0, 300), /await loadSelection\(\)[\s\S]*await loadPreflight\(\)/)
  const panel = read('components/AgentSelectionPanel.vue')
  assert.match(panel, /await loadPlan\(\)/)
  assert.ok(isContractMode(preflightFixture()))
})

// Agent selection helpers: the population is explicit and visible before prepare (run with: npm test)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  applyToEntities,
  decideStep2MountAction,
  filterEntities,
  isSelectionDirty,
  populationWarnings,
  selectionPayload,
  summarizeSelection,
} from '../src/utils/agentSelection.js'

const CATALOG = [
  { id: 'prospect', label: 'Prospect', default_evidence_role: 'target_consumer' },
  { id: 'existing_free', label: 'Existing Free', default_evidence_role: 'influencer_context' },
  { id: 'brand_advocate', label: 'Brand Advocate', default_evidence_role: 'influencer_context' },
  { id: 'brand_platform', label: 'Brand / Platform', default_evidence_role: 'excluded' },
]
const EVIDENCE = [
  { id: 'target_consumer', label: 'Target Consumer' },
  { id: 'influencer_context', label: 'Influencer / Context' },
  { id: 'excluded', label: 'Exclude From Conversion Metrics' },
]

const row = (uuid, name, type, extra = {}) => ({
  entity_uuid: uuid, name, entity_type: type, description: '', active: false,
  simulation_role: 'brand_platform', evidence_role: 'excluded', ...extra,
})

// 10 research consumers + 5 existing users + 3 advocates + 20 platforms
function fixture() {
  const rows = []
  for (let i = 1; i <= 10; i++) rows.push(row(`c${i}`, `C${String(i).padStart(2, '0')} consumer`, 'Consumer'))
  for (let i = 1; i <= 5; i++) rows.push(row(`e${i}`, `Existing user ${i}`, 'ExistingUser'))
  for (let i = 1; i <= 3; i++) rows.push(row(`a${i}`, `Advocate ${i}`, 'Advocate'))
  for (let i = 1; i <= 20; i++) rows.push(row(`p${i}`, `Platform ${i}`, 'Platform'))
  return rows
}

test('the synthetic population selects 18 active agents and 20 context-only entities', () => {
  let rows = fixture()
  const ids = type => rows.filter(r => r.entity_type === type).map(r => r.entity_uuid)
  rows = applyToEntities(rows, ids('Consumer'), { active: true, simulation_role: 'prospect' }, CATALOG)
  rows = applyToEntities(rows, ids('ExistingUser'), { active: true, simulation_role: 'existing_free' }, CATALOG)
  rows = applyToEntities(rows, ids('Advocate'), { active: true, simulation_role: 'brand_advocate' }, CATALOG)

  const summary = summarizeSelection(rows, CATALOG, EVIDENCE)
  assert.equal(summary.active, 18)
  assert.equal(summary.total, 38)
  assert.equal(summary.context, 20)
  assert.deepEqual(summary.bySimulationRole.map(r => [r.id, r.count]), [['prospect', 10], ['existing_free', 5], ['brand_advocate', 3]])
  assert.deepEqual(summary.byEvidenceRole.map(r => r.count), [10, 8, 0])
  assert.deepEqual(summary.byType.Platform, { total: 20, active: 0 })
  assert.deepEqual(populationWarnings(rows), [])
})

test('a population dominated by platform accounts is flagged before preparing', () => {
  let rows = fixture()
  rows = applyToEntities(rows, rows.filter(r => r.entity_type === 'Consumer').map(r => r.entity_uuid),
    { active: true, simulation_role: 'prospect' }, CATALOG)
  rows = applyToEntities(rows, rows.filter(r => r.entity_type === 'Platform').map(r => r.entity_uuid), { active: true }, CATALOG)
  assert.deepEqual(populationWarnings(rows), [{ key: 'nonConsumerMajority', params: { consumers: 10, others: 20 } }])
  assert.deepEqual(populationWarnings(fixture()), [{ key: 'noActive', params: {} }])
})

test('changing a role resets the evidence role unless it is set explicitly', () => {
  const rows = [row('x', 'Google Maps', 'Tool')]
  assert.equal(applyToEntities(rows, ['x'], { simulation_role: 'prospect' }, CATALOG)[0].evidence_role, 'target_consumer')
  const explicit = applyToEntities(rows, ['x'], { simulation_role: 'prospect', evidence_role: 'excluded' }, CATALOG)[0]
  assert.equal(explicit.evidence_role, 'excluded')
  assert.equal(rows[0].evidence_role, 'excluded', 'input is not mutated')
})

test('filters by name, alias, type and active view', () => {
  const rows = [
    row('k', 'Kamptribe', 'Brand', { aliases: ['kamptribe.com'], active: true }),
    row('g', 'Google Maps', 'Tool'),
  ]
  assert.deepEqual(filterEntities(rows, { query: 'kamptribe.COM' }).map(r => r.entity_uuid), ['k'])
  assert.deepEqual(filterEntities(rows, { type: 'Tool' }).map(r => r.entity_uuid), ['g'])
  assert.deepEqual(filterEntities(rows, { view: 'context' }).map(r => r.entity_uuid), ['g'])
  assert.deepEqual(filterEntities(rows, { view: 'active' }).map(r => r.entity_uuid), ['k'])
})

test('active aliases of the same entity are flagged', () => {
  const rows = [
    row('k', 'Kamptribe', 'Brand', { active: true }),
    row('k2', 'KAMPTRIBE', 'Brand', { active: true, duplicate_of: 'k' }),
  ]
  assert.equal(populationWarnings(rows).some(w => w.key === 'activeAliases'), true)
})

test('payload and dirty tracking cover every row', () => {
  const rows = fixture().slice(0, 2)
  const payload = selectionPayload(rows, CATALOG, true)
  assert.equal(payload.confirmed, true)
  assert.equal(payload.entities.length, 2)
  assert.equal(isSelectionDirty(rows, rows), false)
  assert.equal(isSelectionDirty(applyToEntities(rows, ['c1'], { active: true }, CATALOG), rows), true)
})

test('mounting Step 2 never prepares on its own (refresh / navigation)', () => {
  assert.equal(decideStep2MountAction({}), 'select')
  assert.equal(decideStep2MountAction({ prepared: false, selection_confirmed: true }), 'select')
  assert.equal(decideStep2MountAction({ prepared: true, in_sync: true }), 'load_prepared')
  assert.equal(decideStep2MountAction({ prepared: true, legacy_prepared: true }), 'load_prepared')
  assert.equal(decideStep2MountAction({ preparing: true, prepare_task_id: 't1' }), 'resume_prepare')
})

/**
 * Agent selection helpers (pure functions, unit-tested with node --test)
 *
 * A knowledge-graph entity is never automatically a simulation agent: the user
 * marks which entities are active and gives each a simulation role and an
 * evidence role. Only target consumers count in conversion metrics.
 */

export const EVIDENCE_TARGET_CONSUMER = 'target_consumer'

/**
 * Counts shown before prepare/start
 * @param {Array<Object>} entities - selection rows { active, simulation_role, evidence_role, entity_type }
 * @param {Array<Object>} roleCatalog - [{ id, label }]
 * @param {Array<Object>} evidenceRoles - [{ id, label }]
 * @returns {Object} { active, total, context, bySimulationRole, byEvidenceRole, byType }
 */
export function summarizeSelection(entities = [], roleCatalog = [], evidenceRoles = []) {
  const active = entities.filter(e => e.active)
  const roleCounts = {}
  const evidenceCounts = {}
  const byType = {}
  for (const e of entities) {
    const type = e.entity_type || 'Entity'
    byType[type] = byType[type] || { total: 0, active: 0 }
    byType[type].total += 1
    if (e.active) byType[type].active += 1
  }
  for (const e of active) {
    roleCounts[e.simulation_role] = (roleCounts[e.simulation_role] || 0) + 1
    evidenceCounts[e.evidence_role] = (evidenceCounts[e.evidence_role] || 0) + 1
  }
  const roleLabels = Object.fromEntries(roleCatalog.map(r => [r.id, r.label]))
  return {
    active: active.length,
    total: entities.length,
    context: entities.length - active.length,
    bySimulationRole: Object.entries(roleCounts).map(([id, count]) => ({ id, label: roleLabels[id] || id, count })),
    byEvidenceRole: evidenceRoles.map(r => ({ id: r.id, label: r.label, count: evidenceCounts[r.id] || 0 })),
    byType,
  }
}

/**
 * Rows matching the search / type / view filters
 * @param {Array<Object>} entities
 * @param {Object} filters - { query, type, view: 'all' | 'active' | 'context' }
 * @returns {Array<Object>}
 */
export function filterEntities(entities = [], { query = '', type = '', view = 'all' } = {}) {
  const q = query.trim().toLowerCase()
  return entities.filter(e => {
    if (type && (e.entity_type || 'Entity') !== type) return false
    if (view === 'active' && !e.active) return false
    if (view === 'context' && e.active) return false
    if (!q) return true
    return [e.name, e.description, e.entity_type, ...(e.aliases || [])]
      .some(v => (v || '').toLowerCase().includes(q))
  })
}

/**
 * Apply a change to some rows (immutable)
 *
 * Changing the simulation role also resets the evidence role to the role's
 * default unless the patch sets an evidence role explicitly.
 *
 * @param {Array<Object>} entities
 * @param {Iterable<string>} uuids - rows to change
 * @param {Object} patch - { active?, simulation_role?, evidence_role? }
 * @param {Array<Object>} roleCatalog - [{ id, default_evidence_role }]
 * @returns {Array<Object>}
 */
export function applyToEntities(entities = [], uuids = [], patch = {}, roleCatalog = []) {
  const targets = new Set(uuids)
  const defaults = Object.fromEntries(roleCatalog.map(r => [r.id, r.default_evidence_role]))
  return entities.map(e => {
    if (!targets.has(e.entity_uuid)) return e
    const next = { ...e }
    if (patch.active !== undefined) next.active = Boolean(patch.active)
    if (patch.simulation_role) {
      next.simulation_role = patch.simulation_role
      if (!patch.evidence_role && defaults[patch.simulation_role]) {
        next.evidence_role = defaults[patch.simulation_role]
      }
    }
    if (patch.evidence_role) next.evidence_role = patch.evidence_role
    return next
  })
}

/**
 * Request body for PUT /agent-selection
 * @param {Array<Object>} entities
 * @param {Array<Object>} roleCatalog
 * @param {boolean} confirmed
 * @returns {Object}
 */
export function selectionPayload(entities = [], roleCatalog = [], confirmed = false) {
  return {
    confirmed,
    role_catalog: roleCatalog.map(r => ({ id: r.id, label: r.label, default_evidence_role: r.default_evidence_role })),
    entities: entities.map(e => ({
      entity_uuid: e.entity_uuid,
      active: Boolean(e.active),
      simulation_role: e.simulation_role,
      evidence_role: e.evidence_role,
    })),
  }
}

/**
 * Whether the edited rows differ from the saved ones
 * @param {Array<Object>} entities
 * @param {Array<Object>} saved
 * @returns {boolean}
 */
export function isSelectionDirty(entities = [], saved = []) {
  const key = rows => JSON.stringify(rows.map(e => [e.entity_uuid, Boolean(e.active), e.simulation_role, e.evidence_role]))
  return key(entities) !== key(saved)
}

/**
 * Problems the user should see before paying for a simulation
 * @param {Array<Object>} entities
 * @returns {Array<{key: string, params: Object}>} i18n keys under agentSelection.warnings
 */
export function populationWarnings(entities = []) {
  const active = entities.filter(e => e.active)
  const warnings = []
  if (!active.length) {
    warnings.push({ key: 'noActive', params: {} })
    return warnings
  }
  const consumers = active.filter(e => e.evidence_role === EVIDENCE_TARGET_CONSUMER).length
  const others = active.length - consumers
  if (!consumers) warnings.push({ key: 'noTargetConsumers', params: {} })
  else if (others > consumers) warnings.push({ key: 'nonConsumerMajority', params: { consumers, others } })
  const activeIds = new Set(active.map(e => e.entity_uuid))
  const aliases = active.filter(e => e.duplicate_of && activeIds.has(e.duplicate_of))
  if (aliases.length) warnings.push({ key: 'activeAliases', params: { names: aliases.map(e => e.name).join(', ') } })
  const possible = active.filter(e => (e.possible_duplicates || []).some(p => activeIds.has(p.uuid)))
  if (possible.length) warnings.push({ key: 'possibleDuplicatesActive', params: { count: possible.length } })
  return warnings
}

/**
 * Decide what Step 2 does when it mounts (never prepares on its own)
 *
 * - 'load_prepared'   profiles/config exist for the confirmed population: show them
 * - 'resume_prepare'  a preparation is running: poll its task, never start another
 * - 'select'          show the agent selection and wait for the user to confirm
 *
 * @param {Object} population - population block of GET /agent-selection
 * @returns {'load_prepared'|'resume_prepare'|'select'}
 */
export function decideStep2MountAction(population = {}) {
  if (population.preparing && population.prepare_task_id) return 'resume_prepare'
  if (population.prepared) return 'load_prepared'
  return 'select'
}

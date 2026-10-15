/**
 * Run Contract / run preflight helpers (pure functions, unit-tested with node --test)
 *
 * For a contracted simulation the backend preflight (GET /preflight) is the only
 * source of readiness. The UI never derives a green state from local edits, a
 * failed request always reads as NOT READY, and nothing here ever starts or
 * prepares anything on its own.
 */

export const READINESS = {
  NOT_READY: 'not_ready',
  READY_TO_PREPARE: 'ready_to_prepare',
  READY_TO_START: 'ready_to_start',
}

/**
 * Whether a preflight belongs to a simulation with a Run Contract
 * @param {Object} preflight - GET /preflight data
 * @returns {boolean}
 */
export function isContractMode(preflight) {
  return preflight?.contract?.present === true
}

/**
 * State after a preflight fetch: a failed fetch never keeps an earlier green result
 * @param {Object} result - { data?, error? }
 * @returns {{ preflight: Object|null, error: string|null }}
 */
export function preflightFetchState({ data = null, error = null } = {}) {
  if (error || !data) {
    return { preflight: null, error: error ? String(error.message || error) : 'Preflight unavailable' }
  }
  return { preflight: data, error: null }
}

/**
 * Overall readiness shown in the UI (contract mode)
 * @param {Object|null} preflight
 * @param {string|null} error
 * @returns {string} one of READINESS
 */
export function readiness(preflight, error = null) {
  if (error || !preflight) return READINESS.NOT_READY
  if (preflight.ready_to_start === true) return READINESS.READY_TO_START
  if (preflight.ready_to_prepare === true) return READINESS.READY_TO_PREPARE
  return READINESS.NOT_READY
}

/**
 * Whether the explicit Prepare button is enabled
 * @param {Object} input - { preflight, error, preparing, selectionConfirmed }
 * @returns {boolean}
 */
export function canPrepare({ preflight = null, error = null, preparing = false, selectionConfirmed = false } = {}) {
  if (preparing) return false
  if (isContractMode(preflight)) return !error && preflight.ready_to_prepare === true
  return Boolean(selectionConfirmed)
}

/**
 * Whether "Continue to Run" (Step 2 -> Step 3 navigation) is enabled
 * @param {Object} input - { preflight, error, prepared }
 * @returns {boolean}
 */
export function canContinueToRun({ preflight = null, error = null, prepared = false } = {}) {
  if (!prepared) return false
  if (isContractMode(preflight)) return !error && preflight.ready_to_start === true
  return true
}

/**
 * Whether the explicit Start click may call POST /start (after a fresh preflight)
 * @param {Object} input - { preflight, error }
 * @returns {boolean}
 */
export function canStartRun({ preflight = null, error = null } = {}) {
  if (error || !preflight) return false
  if (isContractMode(preflight)) return preflight.ready_to_start === true
  return true
}

/**
 * POST /start body. A contracted run never sends a round count or platform:
 * the backend applies the contract's values.
 * @param {string} simulationId
 * @param {Object|null} preflight
 * @param {Object} options - { maxRounds?, newRun? }
 * @returns {Object}
 */
export function startRequest(simulationId, preflight, { maxRounds = null, newRun = false } = {}) {
  const body = { simulation_id: simulationId }
  if (newRun) body.new_run = true
  if (isContractMode(preflight)) return body
  body.platform = 'parallel'
  if (maxRounds) body.max_rounds = maxRounds
  return body
}

/**
 * Route query for Step 3: contracted runs never transport a round count
 * @param {Object|null} preflight
 * @param {number|null} maxRounds
 * @returns {Object}
 */
export function runRouteQuery(preflight, maxRounds = null) {
  if (isContractMode(preflight) || !maxRounds) return {}
  return { maxRounds }
}

/**
 * What Step 2 does after a selection save / contract apply: re-read the
 * persisted state, never prepare (preparation is its own explicit action)
 * @returns {'reload_state'}
 */
export function afterSelectionSaved() {
  return 'reload_state'
}

/**
 * Locked run length for display, or null when the user may choose
 * @param {Object|null} preflight
 * @returns {number|null}
 */
export function lockedRounds(preflight) {
  return isContractMode(preflight) ? preflight?.rounds?.expected ?? null : null
}

/**
 * Whether the population editor may change rows
 * @param {Object} input - { contractLocked, locked, editing, busy }
 * @returns {boolean}
 */
export function selectionEditable({ contractLocked = false, locked = false, editing = false, busy = false } = {}) {
  if (contractLocked || busy) return false
  return !locked || editing
}

const countRow = (key, value, ok, params = {}) => ({ key, value, ok: Boolean(ok), params })

/**
 * Summary lines of the run preflight (contract mode)
 * @param {Object} preflight
 * @returns {Array<{key: string, value: string, ok: boolean}>} i18n keys under runContract.rows
 */
export function preflightSummaryRows(preflight) {
  if (!isContractMode(preflight)) return []
  const pop = preflight.population || {}
  const roles = preflight.roles || {}
  const plan = preflight.experiment_plan || {}
  const sources = preflight.sources || {}
  const rows = [
    countRow('contract', preflight.contract.id || '?', preflight.contract.valid && preflight.contract.applied),
    countRow('activeRequested', `${pop.active ?? '-'} / ${pop.expected ?? '-'}`, pop.ok),
    countRow('targetConsumers', String(roles.target_consumers ?? '-'),
      roles.target_consumers === roles.expected_target_consumers),
    countRow('influencerContext', String(roles.influencer_context ?? '-'),
      roles.influencer_context === roles.expected_influencer_context),
    countRow('excludedActive', String(roles.excluded_active ?? '-'),
      roles.excluded_active === roles.expected_excluded_active),
    countRow('unexpectedActive',
      (pop.unexpected_active || []).length ? pop.unexpected_active.map(u => u.name).join(', ') : 'none',
      pop.unexpected_active && pop.unexpected_active.length === 0),
    countRow('stages', `${plan.saved_stages ?? '-'} / ${plan.expected_stages ?? '-'}`, plan.ok),
    countRow('rounds', String(preflight.rounds?.expected ?? '-'), preflight.rounds?.ok),
  ]
  for (const doc of sources.documents || []) {
    rows.push(countRow('coverage', doc.ok ? 'PASS' : 'FAIL', doc.ok, { label: doc.label }))
  }
  if (!(sources.documents || []).length) rows.push(countRow('coverage', 'FAIL', false, { label: 'Sources' }))
  const prep = preflight.preparation || {}
  if (prep.prepared || prep.roster_active != null) {
    const expected = prep.expected_agents ?? pop.expected
    rows.push(
      countRow('preparedRoster', `${prep.roster_active ?? '-'} / ${expected}`, prep.roster_matches_contract),
      countRow('twitterProfiles', `${prep.profile_count_twitter ?? '-'} / ${expected}`,
        prep.profile_count_twitter === expected && prep.profile_persona_ids_match),
      countRow('redditProfiles', `${prep.profile_count_reddit ?? '-'} / ${expected}`,
        prep.profile_count_reddit === expected && prep.profile_persona_ids_match),
      countRow('configAgents', `${prep.config_agent_count ?? '-'} / ${expected}`, prep.config_agents_match),
      countRow('selectionRosterContract',
        prep.selection_hash_matches && prep.roster_matches_contract && prep.roster_contract_hash_matches ? 'MATCH' : 'MISMATCH',
        prep.selection_hash_matches && prep.roster_matches_contract && prep.roster_contract_hash_matches),
    )
  }
  return rows
}

/**
 * Per-persona rows (requested id, resolved graph name, roles, status)
 * @param {Object} preflight
 * @returns {Array<Object>}
 */
export function personaRows(preflight) {
  return ((preflight?.population || {}).entities || []).map(e => ({
    personaId: e.persona_id,
    name: e.name || (e.candidates || []).map(c => c.name).join(' | ') || '—',
    simulationRole: e.simulation_role,
    evidenceRole: e.evidence_role,
    active: Boolean(e.selection_active),
    ok: e.status === 'resolved' && e.selection_active
      && e.selection_simulation_role === e.simulation_role && e.selection_evidence_role === e.evidence_role,
    status: e.status,
  }))
}

/**
 * Source coverage matrix rows (one per required section)
 * @param {Object} preflight
 * @returns {Array<{document: string, files: string, section: string, status: string, missingTerms: string, ok: boolean}>}
 */
export function coverageRows(preflight) {
  const rows = []
  for (const doc of preflight?.sources?.documents || []) {
    for (const s of doc.sections || []) {
      rows.push({
        document: doc.label,
        files: (doc.files || []).join(', ') || 'not uploaded',
        section: s.id,
        status: s.status,
        missingTerms: (s.missing_terms || []).join(', '),
        ok: s.status === 'covered' && (doc.files || []).length > 0,
      })
    }
  }
  return rows
}

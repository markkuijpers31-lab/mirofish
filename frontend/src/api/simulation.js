import service, { requestWithRetry } from './index'

/**
 * Create simulation
 * @param {Object} data - { project_id, graph_id?, enable_twitter?, enable_reddit? }
 */
export const createSimulation = (data) => {
  return requestWithRetry(() => service.post('/api/simulation/create', data), 3, 1000)
}

/**
 * Prepare simulation environment (async task)
 * @param {Object} data - { simulation_id, entity_types?, use_llm_for_profiles?, parallel_profile_count?, force_regenerate? }
 */
export const prepareSimulation = (data) => {
  return requestWithRetry(() => service.post('/api/simulation/prepare', data), 3, 1000)
}

/**
 * Get the agent selection: every eligible graph entity with active flag and roles
 * (reading never changes anything)
 * @param {string} simulationId
 * @returns {Promise<Object>} { entities, role_catalog, evidence_roles, summary, confirmed, population }
 */
export const getAgentSelection = (simulationId) => {
  return service.get(`/api/simulation/${simulationId}/agent-selection`)
}

/**
 * Save the agent selection (a changed population invalidates the preparation)
 * @param {string} simulationId
 * @param {Object} data - { entities: [{ entity_uuid, active, simulation_role, evidence_role }], role_catalog?, confirmed }
 * @returns {Promise<Object>} same as getAgentSelection + preparation_invalidated
 */
export const saveAgentSelection = (simulationId, data) => {
  return service.put(`/api/simulation/${simulationId}/agent-selection`, data)
}

/**
 * Get the private experiment plan (stages delivered privately to agents, never posted)
 * @param {string} simulationId
 * @returns {Promise<Object>} { stages, saved, condition_types, completion_types }
 */
export const getExperimentPlan = (simulationId) => {
  return service.get(`/api/simulation/${simulationId}/experiment-plan`)
}

/**
 * Save the private experiment plan (applies to the next run)
 * @param {string} simulationId
 * @param {Object} data - { stages: [...] }
 * @returns {Promise<Object>}
 */
export const saveExperimentPlan = (simulationId, data) => {
  return service.put(`/api/simulation/${simulationId}/experiment-plan`, data)
}

/**
 * Query preparation task progress
 * @param {Object} data - { task_id?, simulation_id? }
 */
export const getPrepareStatus = (data) => {
  return service.post('/api/simulation/prepare/status', data)
}

/**
 * Get simulation status
 * @param {string} simulationId
 */
export const getSimulation = (simulationId) => {
  return service.get(`/api/simulation/${simulationId}`)
}

/**
 * Get simulation Agent Profiles
 * @param {string} simulationId
 * @param {string} platform - 'reddit' | 'twitter'
 */
export const getSimulationProfiles = (simulationId, platform = 'reddit') => {
  return service.get(`/api/simulation/${simulationId}/profiles`, { params: { platform } })
}

/**
 * Get in-progress Agent Profiles in real-time
 * @param {string} simulationId
 * @param {string} platform - 'reddit' | 'twitter'
 */
export const getSimulationProfilesRealtime = (simulationId, platform = 'reddit') => {
  return service.get(`/api/simulation/${simulationId}/profiles/realtime`, { params: { platform } })
}

/**
 * Get simulation config
 * @param {string} simulationId
 */
export const getSimulationConfig = (simulationId) => {
  return service.get(`/api/simulation/${simulationId}/config`)
}

/**
 * Get in-progress simulation config in real-time
 * @param {string} simulationId
 * @returns {Promise} Returns config info including metadata and config content
 */
export const getSimulationConfigRealtime = (simulationId) => {
  return service.get(`/api/simulation/${simulationId}/config/realtime`)
}

/**
 * List all simulations
 * @param {string} projectId - Optional, filter by project ID
 */
export const listSimulations = (projectId) => {
  const params = projectId ? { project_id: projectId } : {}
  return service.get('/api/simulation/list', { params })
}

/**
 * Start simulation (idempotent: an existing run is reported, never replaced)
 * Not retried automatically: starting a run costs LLM budget and must stay an explicit action.
 * @param {Object} data - { simulation_id, platform?, max_rounds?, new_run? }
 *   new_run: true only for an explicit, confirmed restart (creates a NEW run, keeps the old one)
 * @returns {Promise} data: { run_id, run_status, started, already_running, run_exists, ... }
 */
export const startSimulation = (data) => {
  return service.post('/api/simulation/start', data)
}

/**
 * List all runs (executions) of a simulation
 * @param {string} simulationId
 * @returns {Promise} data: { current_run_id, runs: [...] }
 */
export const getSimulationRuns = (simulationId) => {
  return service.get(`/api/simulation/${simulationId}/runs`)
}

/**
 * Stop simulation
 * @param {Object} data - { simulation_id }
 */
export const stopSimulation = (data) => {
  return service.post('/api/simulation/stop', data)
}

/**
 * Get simulation run real-time status (includes persistent run_id / run_status)
 * @param {string} simulationId
 * @param {string} [runId] - Optional, default: current run
 */
export const getRunStatus = (simulationId, runId) => {
  const params = runId ? { run_id: runId } : {}
  return service.get(`/api/simulation/${simulationId}/run-status`, { params })
}

/**
 * Get simulation run detailed status (including recent actions)
 * @param {string} simulationId
 * @param {string} [runId] - Optional, default: current run
 */
export const getRunStatusDetail = (simulationId, runId) => {
  const params = runId ? { run_id: runId } : {}
  return service.get(`/api/simulation/${simulationId}/run-status/detail`, { params })
}

/**
 * Get posts in simulation
 * @param {string} simulationId
 * @param {string} platform - 'reddit' | 'twitter'
 * @param {number} limit - Return count
 * @param {number} offset - Offset
 */
export const getSimulationPosts = (simulationId, platform = 'reddit', limit = 50, offset = 0) => {
  return service.get(`/api/simulation/${simulationId}/posts`, {
    params: { platform, limit, offset }
  })
}

/**
 * Get simulation timeline (grouped by round)
 * @param {string} simulationId
 * @param {number} startRound - Start round
 * @param {number} endRound - End round
 */
export const getSimulationTimeline = (simulationId, startRound = 0, endRound = null) => {
  const params = { start_round: startRound }
  if (endRound !== null) {
    params.end_round = endRound
  }
  return service.get(`/api/simulation/${simulationId}/timeline`, { params })
}

/**
 * Get Agent statistics
 * @param {string} simulationId
 */
export const getAgentStats = (simulationId) => {
  return service.get(`/api/simulation/${simulationId}/agent-stats`)
}

/**
 * Get simulation action history
 * @param {string} simulationId
 * @param {Object} params - { limit, offset, platform, agent_id, round_num }
 */
export const getSimulationActions = (simulationId, params = {}) => {
  return service.get(`/api/simulation/${simulationId}/actions`, { params })
}

/**
 * Close simulation environment (graceful exit)
 * @param {Object} data - { simulation_id, timeout? }
 */
export const closeSimulationEnv = (data) => {
  return service.post('/api/simulation/close-env', data)
}

/**
 * Get simulation environment status
 * @param {Object} data - { simulation_id }
 */
export const getEnvStatus = (data) => {
  return service.post('/api/simulation/env-status', data)
}

/**
 * Batch interview Agents
 * @param {Object} data - { simulation_id, interviews: [{ agent_id, prompt }] }
 */
export const interviewAgents = (data) => {
  return requestWithRetry(() => service.post('/api/simulation/interview/batch', data), 3, 1000)
}

/**
 * Get historical simulation list (with project details)
 * For home page historical project display
 * @param {number} limit - Return count limit
 */
export const getSimulationHistory = (limit = 20) => {
  return service.get('/api/simulation/history', { params: { limit } })
}


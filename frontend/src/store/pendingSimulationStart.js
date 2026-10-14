/**
 * One-shot, in-memory flag set by the explicit "Start simulation" click in Step 2
 * Step 3 consumes it once on mount. It is not persisted, so a page reload or
 * browser navigation can never trigger a simulation start.
 */
import { reactive } from 'vue'

const state = reactive({
  simulationId: null,
})

export function requestSimulationStart(simulationId) {
  state.simulationId = simulationId
}

/**
 * Return true (once) if a start was explicitly requested for this simulation
 * @param {string} simulationId
 * @returns {boolean}
 */
export function consumeSimulationStartRequest(simulationId) {
  const requested = !!simulationId && state.simulationId === simulationId
  state.simulationId = null
  return requested
}

export default state

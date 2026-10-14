/**
 * Simulation run lifecycle helpers (pure functions, unit-tested with node --test)
 *
 * A run is one execution of a simulation. Its persistent status comes from the
 * backend (run-status: run_status). Page mounts, reloads and navigation must
 * never start a run; only an explicit user action may.
 */

export const RUN_STATUS = {
  NOT_STARTED: 'not_started',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
  STOPPED: 'stopped',
  REPORTING: 'reporting',
  REPORT_COMPLETED: 'report_completed',
}

const FINISHED = new Set([
  RUN_STATUS.COMPLETED,
  RUN_STATUS.FAILED,
  RUN_STATUS.STOPPED,
  RUN_STATUS.REPORTING,
  RUN_STATUS.REPORT_COMPLETED,
])

const REPORTABLE = new Set([
  RUN_STATUS.COMPLETED,
  RUN_STATUS.STOPPED,
  RUN_STATUS.FAILED,
  RUN_STATUS.REPORTING,
  RUN_STATUS.REPORT_COMPLETED,
])

/**
 * Whether a run holds finished evidence
 * @param {string} runStatus
 * @returns {boolean}
 */
export function isRunFinished(runStatus) {
  return FINISHED.has(runStatus)
}

/**
 * Whether a report can be generated from a run
 * @param {string} runStatus
 * @returns {boolean}
 */
export function canGenerateReport(runStatus) {
  return REPORTABLE.has(runStatus)
}

/**
 * Decide what Step 3 does when it mounts
 *
 * - 'attach'         a run is active: poll it, never start another one
 * - 'show_finished'  a run already finished: show it, never restart it
 * - 'start'          no run exists AND the user just clicked "Start" in Step 2
 * - 'show_idle'      no run exists and nothing was requested (reload, navigation):
 *                    show a Start button and wait for the user
 *
 * @param {Object} input
 * @param {string} [input.runStatus] - run_status from the backend (missing = no run)
 * @param {boolean} [input.pendingStart] - true only right after the explicit Step 2 click
 * @returns {'attach'|'show_finished'|'start'|'show_idle'}
 */
export function decideStep3MountAction({ runStatus, pendingStart = false } = {}) {
  if (runStatus === RUN_STATUS.RUNNING) return 'attach'
  if (isRunFinished(runStatus)) return 'show_finished'
  if (pendingStart === true) return 'start'
  return 'show_idle'
}

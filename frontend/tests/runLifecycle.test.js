// Regression tests for the Step 3 run lifecycle (run with: npm test)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  RUN_STATUS,
  decideStep3MountAction,
  isRunFinished,
  canGenerateReport,
} from '../src/utils/runLifecycle.js'

test('page reload of a completed run shows it and never starts a new simulation', () => {
  assert.equal(decideStep3MountAction({ runStatus: RUN_STATUS.COMPLETED }), 'show_finished')
  // even if a stale start request were present, a finished run is never restarted
  assert.equal(decideStep3MountAction({ runStatus: RUN_STATUS.COMPLETED, pendingStart: true }), 'show_finished')
})

test('every finished status is shown, not restarted', () => {
  for (const status of [
    RUN_STATUS.COMPLETED,
    RUN_STATUS.FAILED,
    RUN_STATUS.STOPPED,
    RUN_STATUS.REPORTING,
    RUN_STATUS.REPORT_COMPLETED,
  ]) {
    assert.equal(decideStep3MountAction({ runStatus: status, pendingStart: true }), 'show_finished', status)
    assert.equal(isRunFinished(status), true, status)
  }
})

test('reopening Step 3 while a run is active attaches to it', () => {
  assert.equal(decideStep3MountAction({ runStatus: RUN_STATUS.RUNNING }), 'attach')
  assert.equal(decideStep3MountAction({ runStatus: RUN_STATUS.RUNNING, pendingStart: true }), 'attach')
})

test('a run is only started after the explicit Step 2 click', () => {
  assert.equal(decideStep3MountAction({ runStatus: undefined, pendingStart: true }), 'start')
  assert.equal(decideStep3MountAction({ runStatus: RUN_STATUS.NOT_STARTED, pendingStart: true }), 'start')
})

test('reload or navigation without an explicit click never starts a run', () => {
  assert.equal(decideStep3MountAction({ runStatus: undefined }), 'show_idle')
  assert.equal(decideStep3MountAction({ runStatus: RUN_STATUS.NOT_STARTED }), 'show_idle')
  assert.equal(decideStep3MountAction({}), 'show_idle')
  assert.equal(decideStep3MountAction(), 'show_idle')
})

test('reports can be generated from finished runs only', () => {
  assert.equal(canGenerateReport(RUN_STATUS.COMPLETED), true)
  assert.equal(canGenerateReport(RUN_STATUS.REPORT_COMPLETED), true)
  assert.equal(canGenerateReport(RUN_STATUS.RUNNING), false)
  assert.equal(canGenerateReport(RUN_STATUS.NOT_STARTED), false)
})

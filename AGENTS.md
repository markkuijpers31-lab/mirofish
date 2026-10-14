# AGENTS.md

Guide for coding agents operating in this repository. MiroFish is a multi-agent
AI prediction engine: Flask backend (`backend/`) + Vue 3 frontend (`frontend/`)
+ shared i18n (`locales/`). Monorepo orchestrated by root `package.json` via
`concurrently`. Code comments/docstrings are **English**; identifiers
are English. License: AGPL-3.0.

## Build / Run / Test Commands

All commands run from repo root unless noted. Node >=18, Python >=3.11 <=3.12,
`uv` for Python packages.

```bash
# Install everything (root + frontend npm + backend uv venv)
npm run setup:all

# Dev (both services, concurrent, --kill-others)
npm run dev
npm run backend   # cd backend && uv run python run.py  (Flask :5001)
npm run frontend  # cd frontend && npm run dev          (Vite  :3000)

# Production build (frontend only)
npm run build

# Backend deps only
npm run setup:backend   # cd backend && uv sync

# Docker
docker compose up -d    # reads root .env, ports 3000+5001, mounts backend/uploads
```

### Tests

Backend tests live in `backend/tests/` (pytest, `conftest.py` provides an
isolated `uploads` fixture, a fake OASIS process and simulation/project helpers).
Frontend has a dependency-free `node --test` suite in `frontend/tests/` for pure
helpers (e.g. `src/utils/runLifecycle.js`).

```bash
cd backend && uv run pytest                       # all
cd backend && uv run pytest path/to/test_file.py  # single file
cd backend && uv run pytest path/to/test_file.py::test_name  # single test
cd frontend && npm test                           # frontend unit tests
```
The full suite must stay green. `tests/test_e2e_pipeline.py` runs the whole
pipeline (upload → ontology → local graph → prepare → fake OASIS run → report →
ZIP export) without any external service; `conftest.py` blocks real LLM and
embedding calls.
`backend/scripts/test_profile_format.py` is a **manual print-based script**, not
a pytest test — run it with `uv run python scripts/test_profile_format.py`.

### Lint / Format

**None configured.** No ruff/black/flake8/mypy/isort on the backend; no
eslint/prettier on the frontend. No CI lint workflow (only Docker image build).
Match the existing de-facto style described below; do not add a linter unless
asked.

## Backend Conventions (Python / Flask)

**Package layout:** `backend/app/` with `api/` (blueprints), `models/`
(dataclasses), `services/` (business logic), `utils/` (logger, retry, llm_client,
locale). Entry point `backend/run.py`. Each `__init__.py` is a manifest: docstring
+ relative imports + `__all__`.

**Imports:** Relative for intra-package (`from .config import Config`,
`from ..services.x import Y`). Absolute `from app...` only in `run.py` and
`scripts/` (which shim `sys.path`). Group: stdlib → third-party → local, blank-line
separated. Late imports inside functions are acceptable for optional/heavy deps.

**Naming:** `snake_case` modules/functions, `PascalCase` classes, `UPPER_SNAKE`
constants, `_`-prefixed private helpers. Blueprints: `snake_case_bp`
(`graph_bp`, `simulation_bp`, `report_bp`). String IDs prefixed
(`proj_`, `sim_`, `mirofish_`).

**Models:** `@dataclass` + `(str, Enum)` for status enums. Hand-written
`to_dict()` / `from_dict()` on every model. **No app-defined Pydantic models** —
pydantic is only used in generated ontology code templates and the OASIS SDK.

**Type hints:** Annotate function signatures (`typing.Dict/Any/List/Optional/
Callable/Tuple`). Locals usually unannotated. Be consistent with neighbors.

**Errors:** No custom exception classes; raise `ValueError` for bad state. Every
Flask route wraps in `try/except Exception as e:` returning the uniform envelope:
```python
except Exception as e:
    return jsonify({"success": False, "error": str(e),
                    "traceback": traceback.format_exc()}), 500
```
Success: `{"success": True, "data": {...}}` (often + `"count"`, `"message"`).
`ValueError` → 400/404. Background threads catch exceptions and flip the
`TaskManager`/`Project` status to `FAILED`.

**Logging:** `from ..utils.logger import get_logger` then
`logger = get_logger('mirofish.<area>')` at module top (areas: `api`,
`api.simulation`, `build`, `request`, `retry`, `simulation`, etc.).
f-string messages. Do **not** use `logging.getLogger(__name__)` —
`ontology_generator.py` does this and is the one known inconsistency. Rotating
file handler writes to `backend/logs/<date>.log`.

**Async:** Flask is sync (`threaded=True`). Long work goes to
`threading.Thread(target=..., daemon=True)`. **Background threads must re-call
`set_locale()`** — locale is thread-local; capture `get_locale()` before spawning
and restore it inside the thread. Retry helpers in `utils/retry.py`:
`retry_with_backoff` (sync), `retry_with_backoff_async` (async),
`RetryableAPIClient`.

**Config:** `app/config.py` `Config` class loads root `.env` via
`dotenv.load_dotenv('../../.env')` (relative to `config.py`). Required:
`LLM_API_KEY`. Optional: `LLM_BASE_URL`, `LLM_MODEL_NAME`,
`LLM_BOOST_*`, `FLASK_HOST/PORT/DEBUG`, `OASIS_DEFAULT_MAX_ROUNDS`,
`REPORT_AGENT_*`, `SECRET_KEY`. `Config.validate()` is called in `run.py` before
serving. Frontend env: `VITE_API_BASE_URL` (default `http://localhost:5001`).

**Docstrings:** Module docstring (English, one line) at top of every `.py`.
Functions/classes: English summary, Google-style `Args:`/`Returns:`. Flask routes
embed request/response JSON shapes as literal blocks — treat these as the API docs.

**Windows:** `run.py` applies a UTF-8 stdout fix before other imports. Keep that
block first.

## Frontend Conventions (Vue 3 + Vite, plain JS)

**SFCs:** `<script setup>` Composition API only. No Options API, no
`defineComponent({})`. `defineProps({...object syntax...})`,
`defineEmits([...])`. Use `ref`, `reactive`, `computed`, `watch`, `useRouter()`,
`useI18n()`.

**Style:** No semicolons. 2-space indent. Single quotes. Trailing commas in
multi-line objects/arrays. Plain JS (no TypeScript).

**Imports:** Vue ecosystem first (`vue`, `vue-router`, `vue-i18n`), then local
modules. Relative paths dominate (`../api/simulation`); `@` → `src` and
`@locales` → root `locales/` aliases exist in `vite.config.js` and may be used.

**Naming:** `PascalCase.vue` files (views suffixed `*View.vue`; step components
`Step1..Step5`). `camelCase` JS vars/functions/refs. `kebab-case` template tags
and CSS classes. Route names `PascalCase`.

**State:** No Pinia/Vuex. `store/` holds `reactive()` singletons exporting
setter/getter/clear functions. Component-local state via `ref`/`reactive`.

**API:** Single `axios.create()` instance in `src/api/index.js` (`baseURL` from
`VITE_API_BASE_URL || 'http://localhost:5001'`, 5min timeout). Request interceptor
injects `Accept-Language` from `i18n.global.locale.value`; response interceptor
checks the `success` envelope and rejects on failure. `requestWithRetry(fn,
maxRetries=3, delay=1000)` with exponential backoff. Per-resource modules
(`graph.js`, `simulation.js`, `report.js`) export named functions. JSDoc
`@param`/`@returns` on each.

**i18n:** `vue-i18n` `legacy: false`, `useI18n()`, `$t()`, `<i18n-t>` for rich
inline text. Messages live in **root `locales/*.json`** (`en.json`, `zh.json`),
shared with the backend's `utils/locale.py` (only `en.json` exists in this fork).
Persisted locale in `localStorage` (key `locale`, default `en`).

**CSS:** Plain CSS, no preprocessor. `<style scoped>` default on components;
`App.vue` has unscoped global resets. Global font: `'JetBrains Mono', 'Space
Grotesk', 'Noto Sans SC', monospace`. Palette: black/white + orange
(`#FF4500`/`#FF5722`).

## Simulation Runs, Evidence and Reports

- A simulation (`uploads/simulations/<sim_id>/`) is configuration (state, config,
  profiles). Every execution is a **run** with its own immutable directory
  `uploads/simulations/<sim_id>/runs/<run_id>/` (`services/run_store.py`): config +
  profile snapshot, `personas.json`, `sources/`, `assessment/brief.md`,
  `twitter|reddit/actions.jsonl`, `simulation.log`, DBs, IPC, `run.json`.
- Run status (`run.json`): `not_started → running → completed|failed|stopped →
  reporting → report_completed`. `POST /api/simulation/start` is idempotent; only
  `new_run: true` (explicit, confirmed Restart) creates another run. Never delete or
  append to another run's files; `force` is ignored.
- Simulations from before runs existed are exposed read-only as run `legacy`.
- Reports read the run's raw evidence directly (`services/run_evidence.py`); the
  local graph is supplementary. Fixed evidence sections are built in
  `services/report_evidence_sections.py` and validated in code (uncited claims →
  missing). `GET /api/report/<id>/export` returns the audit ZIP.
- Log run-scoped messages with `get_run_logger(name, simulation_id, run_id, area)`
  so every line carries `[sim][run][area]`.
- Step 3 must never start a run on mount; see `frontend/src/utils/runLifecycle.js`.

## Agent Selection, Private Instructions and Action Origin

- **Graph entity != agent.** `services/agent_selection.py`: the user confirms which
  eligible graph entities are active (`agent_selection.json`, per entity: active,
  `simulation_role` from an editable role catalog, `evidence_role` ∈
  `target_consumer | influencer_context | excluded`). `/prepare` refuses without a
  confirmed selection; prepare freezes it into `agent_roster.json` (active agents get
  ids 0..N-1). Inactive entities stay in the graph as context and never get profiles.
  Changing the population invalidates the preparation; existing runs keep their snapshot.
- Duplicates: `services/entity_canonicalizer.py` groups high-confidence variants
  (case/domain/persona id) as aliases and only flags ambiguous short forms. Never merge graph nodes.
- The assessment brief is research design, not simulation content
  (`services/research_instructions.py`): config/persona LLM context excludes brief
  documents and private instruction text; seed posts that look like instructions are
  rejected (`event_config.rejected_seed_posts`).
- Private instructions/stages: `experiment_plan.json` (default: derived from the brief)
  is resolved per run into `runs/<run_id>/experiment_plan.json`. The runner
  (`scripts/run_orchestration.py`) delivers stage instructions into agent memory only,
  never as posts. Trigger round >= 1; round 0 is seed content only.
- Every logged action has `origin`: `agent_generated | system_seed |
  researcher_instruction` (events use `orchestration`). Only `agent_generated` is
  behavioural evidence; conversion metrics default to
  `evidence_role = target_consumer AND origin = agent_generated` (`RunEvidence.consumer_metrics`).
- Required sources (`services/source_requirements.py`): a simulation may declare authoritative
  documents with anchored sections (`## Pricing {#pricing}`) and required terms; `/prepare` and
  `/start` refuse until covered. Reports pass authoritative documents in full; stage
  `context_sections` pull verbatim sections into private stage context (not leak-guarded).
- Stage plan extras: `authored_mentioned` / `authored_not_mentioned` conditions (own text only),
  `records` (orchestration facts such as EXPOSED, decided from deliveries), `derived_fields`
  (decided from the complete log), `seed_blocked_terms`. A saved stage reaching no agent, or a
  missing context section, refuses the run. Assessment-brief documents never enter the graph.

## Ports / Proxy

Backend `5001`, frontend dev `3000`. Vite proxies `/api` → `http://localhost:5001`
(`changeOrigin`, `secure: false`). Note: `api/index.js` defaults to an absolute
`baseURL`, so set `VITE_API_BASE_URL=''` to use the proxy.

## Known Inconsistencies (fix when touching the file)

- Frontend mixes `export function foo()` and `export const foo = () => {}`.
- `traceback.format_exc()` is leaked in 500 responses (debug aid; not prod-safe).
- `scripts/test_profile_format.py` is named like a pytest test but is a manual
  print-based script.

## Agent Workflow Notes

- Match the style of neighboring files; this repo has no linter to enforce.
- Do not add dependencies (Python or JS) for what a few lines can do.
- Keep the `{success, data}` / `{success, error}` envelope on every new route.
- Re-set locale in any new background thread.
- Don't commit `.env`; the root `.env` is gitignored and holds live keys.
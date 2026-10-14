# First real end-to-end test on Coolify

A checklist for one safe, low-cost end-to-end run of the Zep-free MiroFish
pipeline (local SQLite graph, immutable runs, raw-evidence report).

Replace `<backend>` with the backend URL (port 5001) and `<sim>`, `<run>`,
`<project>`, `<graph>`, `<report>` with the IDs shown in the UI or logs.

## 0. Before deploying

- The branch's backend suite must be green: `cd backend && uv run pytest`.
  It also runs a synthetic pipeline test (`tests/test_e2e_pipeline.py`).
- `.env` needs `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME`.
  No `ZEP_API_KEY` or other graph service is used.

## 1. Deploy

1. Deploy the branch in Coolify.
2. Check that **persistent storage** is mounted at `/app/backend/uploads`. All of
   these live under it: projects and sources, `graphs/<graph>/graph.db`,
   `simulations/<sim>/runs/<run>/` and `reports/<report>/`.
3. Do a hard refresh in the browser (Ctrl+Shift+R) so no old frontend bundle is
   cached.
4. Check `https://<backend>/health`.

## 2. Upload sources and generate the ontology

1. Upload one PDF, one MD and one TXT. Put the assessment instructions either in
   the requirement field or in a file with `assessment` or `brief` in its name.
   They should contain:
   - persona IDs, e.g. "C01-C10";
   - the funnel fields as a bullet list under a "for each persona" line;
   - hypotheses as `H1: …`.
2. Optionally fill in "additional context". It is now stored with the project and
   is part of the brief.
3. Verify with `GET https://<backend>/api/graph/project/<project>`:
   - `files[]` lists every original filename with `extraction_status: "ok"` and a
     `text_length` > 0.
   - A file with `empty` or `failed` is shown explicitly; it no longer breaks the
     upload. A scanned PDF without a text layer shows up as `empty`.
   - `ontology.entity_types` ends with `Person` and `Organization`.
   - If the LLM returns a malformed ontology, it is normalised. If the LLM call
     itself fails, the project gets `status: failed` with an `error`.

## 3. Build the graph

1. Click build.
2. The task must end `completed`, with a `node_count` > 0.
3. If no entity could be extracted, the build now fails with "No entities could be
   extracted …". Fix the LLM settings instead of continuing.
4. Optional: in `GET https://<backend>/api/graph/data/<graph>` check that name
   variants such as "C01" and "C01 Dutch first-time family" appear as **one** node.
   The longest name is kept and the other is listed in `attributes.aliases`.

## 4. Prepare (Step 2)

1. Choose a **small number of rounds** to keep the cost low.
2. Wait until preparation is complete.
3. Agent names equal the graph entity names. Usernames are deterministic, and the
   country is never invented: it is "Unknown" when the sources do not state it.
4. If preparation fails, Step 2 shows the error. The simulation cannot be started
   until a new prepare succeeds, and earlier runs are not touched.

## 5. Run (Step 3)

1. Click Start.
2. During the run, in the Coolify terminal:
   `ls /app/backend/uploads/simulations/<sim>/runs/` must show exactly one
   `run_…` directory.
3. When it finishes, press **F5** in Step 3, then browse to the report and back.
   - The badge shows "COMPLETED · run_…".
   - No new directory appears under `runs/`.
   - The logs show no new "Run created" line.

## 6. Report

1. Click Generate report.
2. In the Coolify logs, filter on `[<sim>][<run>][report]`. You should see:
   - `Action evidence loaded: twitter=… reddit=…`
   - `Source files loaded`
   - `Graph retrieval …`
   - `Report completed`
3. In the report, check:
   - every requested persona (C01–C10) has its own block;
   - missing stages are shown as MISSING;
   - requested IDs without an agent are marked "Not present in the simulation";
   - there is a hypothesis table;
   - the evidence appendix lists all source files with "used in this report".
4. The action counts in the appendix should match
   `wc -l runs/<run>/*/actions.jsonl` minus the event lines (`simulation_start`,
   `round_*`, `simulation_end`).

## 7. Audit bundle

Click **Download run bundle (ZIP)** and check that it contains:

- `MANIFEST.json`
- `run/`, `config/`
- `personas/`
- `sources/` (a manifest, text and the originals)
- `assessment/` (`brief.md` and `spec.json`)
- `actions/`
- `evidence/normalized_evidence.json`
- `report/` (Markdown and JSON, `agent_log.jsonl`, `console_log.txt`)

## 8. Optional: restart (incurs new simulation cost)

1. Click **Restart simulation (new run)** and confirm.
2. A second `run_…` directory appears under `runs/`.
3. Running `sha256sum` over the files of the first run must give unchanged results.

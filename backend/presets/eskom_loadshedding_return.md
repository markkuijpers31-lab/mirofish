# Eskom Loadshedding Return — Public Sentiment & Social Unrest Prediction

> **MiroFish Simulation Requirement Preset**
> Version: v1.0.0 | Created: 2026-07-22

---

## Project Name

Eskom Loadshedding Return Sentiment Prediction

## Simulation Requirement

Eskom, South Africa's state-owned power utility, announces the return of
loadshedding after a prolonged period of uninterrupted power supply. The
announcement cites multiple generation unit failures at Kusile, Medupi, and
Kriel power stations, delayed maintenance backlogs, and depleted emergency
reserves. Loadshedding is implemented at Stage 3 (rotational power cuts of
~4 hours, three times per day) effective immediately, with warnings that
higher stages may follow within 72 hours if generation capacity is not
restored. The announcement comes just weeks after government officials
publicly declared that loadshedding was "a thing of the past."

Predict the public sentiment response and potential social unrest dynamics
across social media platforms (Twitter and Reddit) in the 72 hours following
the announcement. Consider the perspectives of ordinary South African
citizens across different economic strata, small business owners facing
disruption costs, large industrial power users, political parties (ANC, DA,
EFF, ActionSA), energy analysts and commentators, Eskom employees and
engineers, renewable energy advocates, taxi industry operators, restaurant
and hospitality owners, medical facility administrators concerned about
backup generator costs, students preparing for exams, government officials
defending the grid management, opposition politicians calling for
privatization, solar/installation companies seeing demand spikes, and
neighbouring country observers (Zimbabwe, Mozambique) watching for cross-
border impacts. How does sentiment evolve from the initial shock through
anger, blame attribution, adaptation, and political mobilization over a
72-hour period?

## Additional Context

Key factors to model:

1. **Broken promises** — The return contradicts recent government "end of loadshedding" declarations, amplifying public anger and eroding trust
2. **Economic strata divide** — Wealthy households with solar/inverters experience minimal impact; lower-income households bear disproportionate burden
3. **Small business impact** — SMEs face spoilage (refrigeration), lost trading hours, and generator fuel costs with no compensation
4. **Political opportunism** — Opposition parties use the crisis to attack the ruling ANC; ANC officials deflect blame to legacy issues
5. **Stage progression anxiety** — Fear that Stage 3 will escalate to Stage 6 (12+ hours off per day) as has happened historically
6. **Renewable energy momentum** — Solar installation companies report demand surges; advocates push for accelerated decentralization
7. **Eskom internal dynamics** — Employees face public hostility; engineers cite underfunding and aging infrastructure
8. **Cross-border effects** — South Africa imports/export power from Mozambique (Cahora Bassa) and trade impacts with Zimbabwe
9. **Media framing** — Outlets like News24, EWN, and TimesLive shape narrative; international media (Reuters, BBC) frame as "South Africa's chronic crisis"
10. **Historical fatigue** — Citizens have lived through 15+ years of loadshedding; collective trauma and "dark humour" coping mechanisms
11. **Schedule anxiety** — The EskomSePush app and loadshedding schedules become focal points of daily planning
12. **Crime and safety** — Power cuts correlate with increased incidents of burglary, traffic congestion, and traffic light failures

The simulation should capture the initial announcement shock, political
blame-shifting, citizen anger and dark humour, business adaptation, solar
industry opportunism, analyst commentary, and the evolution of collective
sentiment from outrage to resigned adaptation.

## Recommended Entity Types

Suggested agent personas for the knowledge graph:

| Entity Type | Description |
|---|---|
| Citizen | Ordinary South African citizen (varied income levels) |
| SmallBusinessOwner | SME owner (restaurant, shop, salon, mechanic) |
| IndustrialPowerUser | Large factory/mining operation power consumer |
| Politician | Political party representative (ANC, DA, EFF, ActionSA) |
| EnergyAnalyst | Energy sector commentator or analyst |
| EskomEmployee | Eskom staff member (engineer, spokesperson, technician) |
| RenewableEnergyAdvocate | Pro-solar/pro-renewables activist or entrepreneur |
| Journalist | News reporter covering the story (News24, EWN, SABC) |
| TaxiOperator | Minibus taxi driver or association member |
| MedicalAdmin | Hospital/clinic administrator managing backup power |
| Student | Learner or university student affected by study disruption |
| SolarCompany | Solar/inverter installation company |
| GovernmentOfficial | Municipal or national government official |
| NeighbouringObserver | Citizen of Zimbabwe/Mozambique watching cross-border impact |
| SocialMediaInfluencer | Influencer or comedian shaping online discourse |
| Economist | Economist commenting on GDP and investment impact |

## Recommended Platforms

| Platform | Enabled |
|---|---|
| Twitter | Yes (primary — real-time reactions, political discourse) |
| Reddit | Yes (r/southafrica — longer-form analysis, community discussion) |

## Recommended Time Configuration

| Parameter | Value |
|---|---|
| Total simulation hours | 72 |
| Minutes per round | 30 |
| Approx. total rounds | 144 |

## Suggested Source Documents

Upload these (or similar) documents when creating the MiroFish project so the
ontology generator and knowledge graph have rich source material:

1. Eskom official loadshedding announcement and system status report (PDF/MD)
2. Recent government statements declaring "end of loadshedding" (MD)
3. News articles covering the announcement (News24, EWN, TimesLive) (MD/TXT)
4. EskomSePush notification or schedule example (TXT)
5. Historical loadshedding impact data (cost to economy, GDP impact studies) (PDF/MD)
6. Political party statements on energy policy (ANC, DA, EFF) (MD)
7. Renewable energy market report for South Africa (PDF/MD)
8. Social media posts/screenshots from prior loadshedding events (TXT)

## How to Use This Preset

### Option A — Web UI (Recommended for first run)

1. **Start MiroFish**
   ```bash
   npm run dev
   ```
   Open `http://localhost:3000` in your browser.

2. **Create project** — Enter **Project Name**: `Eskom Loadshedding Return Sentiment Prediction`. Paste the **Simulation Requirement** and **Additional Context** into the project creation form.

3. **Upload documents** — Upload any of the suggested source documents above (even 1-2 is enough to start).

4. **Generate ontology** — Click to auto-generate entity/edge types from the documents and requirement. The system should detect entities like `Citizen`, `EskomEmployee`, `Politician`, etc.

5. **Build graph** — The knowledge graph is constructed from the uploaded documents. Review the extracted entities and relationships.

6. **Generate profiles** — Agent personas are created for each entity in the graph. Each agent gets a name, bio, persona description, and behavioural traits.

7. **Run simulation** — Twitter + Reddit simulation runs for 72 simulated hours (~144 rounds). Watch agents post, reply, argue, and spread narratives in real time.

8. **Generate report** — The Report Agent produces a structured prediction report covering:
   - Likely emotional trajectory (shock → anger → blame → dark humour → resignation → adaptation)
   - Main objections and counter-narratives
   - Messages most likely to go viral
   - Weak assumptions in the scenario
   - Political mobilization signals
   - Business adaptation patterns
   - Concrete actions that could improve the outcome

9. **Interact** — After the report, ask follow-up questions like:
   - "What if Eskom escalates to Stage 6 on day 2?"
   - "Interview a simulated small business owner about their biggest concern"
   - "What single government action would most reduce public anger?"
   - "How does sentiment differ between wealthy and lower-income agents?"

### Option B — API (for automation)

```bash
# 1. Create project with ontology generation
curl -X POST http://localhost:5001/api/graph/ontology/generate \
  -F "project_name=Eskom Loadshedding Return Sentiment Prediction" \
  -F "simulation_requirement=Eskom announces the return of loadshedding at Stage 3 after weeks of government declarations that it was over. Predict public sentiment across Twitter and Reddit over 72 hours. Consider citizens, small business owners, politicians (ANC/DA/EFF), Eskom employees, renewable energy advocates, energy analysts, journalists, taxi operators, medical admins, students, solar companies, and neighbouring country observers." \
  -F "additional_context=Key factors: (1) Broken government promises amplify anger; (2) Economic strata divide — wealthy with solar vs lower-income bearing burden; (3) Small business spoilage and generator costs; (4) Political opportunism from opposition; (5) Fear of escalation to Stage 6; (6) Solar industry demand surge; (7) 15+ years of collective fatigue and dark humour; (8) EskomSePush app as daily planning tool; (9) Crime and safety during outages; (10) Cross-border power trade impacts." \
  -F "files=@eskom_announcement.md" \
  -F "files=@news_coverage.md"

# Response contains project_id — use for subsequent steps:

# 2. Build knowledge graph
curl -X POST http://localhost:5001/api/graph/build \
  -H "Content-Type: application/json" \
  -d '{"project_id": "<PROJECT_ID>"}'

# 3. Check graph build status (poll until complete)
curl http://localhost:5001/api/graph/tasks

# 4. Create simulation
curl -X POST http://localhost:5001/api/simulation/create \
  -H "Content-Type: application/json" \
  -d '{"graph_id": "<GRAPH_ID>", "requirement": "..."}'

# 5. Prepare simulation (generate profiles + config)
curl -X POST http://localhost:5001/api/simulation/prepare \
  -H "Content-Type: application/json" \
  -d '{"simulation_id": "<SIM_ID>"}'

# 6. Start simulation
curl -X POST http://localhost:5001/api/simulation/start \
  -H "Content-Type: application/json" \
  -d '{"simulation_id": "<SIM_ID>"}'

# 7. Generate report (after simulation completes)
curl -X POST http://localhost:5001/api/report/generate \
  -H "Content-Type: application/json" \
  -d '{"simulation_id": "<SIM_ID>"}'
```

## Expected Simulation Outcomes

Based on the scenario, the simulation should reveal:

| Phase | Time | Expected Sentiment |
|---|---|---|
| Shock & disbelief | Hours 0-6 | "Not again", "We were promised this was over" |
| Anger & blame | Hours 6-24 | Political finger-pointing, Eskom criticism, government distrust |
| Dark humour & coping | Hours 12-48 | Memes, "EskomSePush check", gallows humour, nostalgic jokes |
| Adaptation | Hours 24-60 | Loadshedding schedules shared, solar quotes requested, generator purchases |
| Political mobilization | Hours 24-72 | Opposition calls for privatization, protests, petition campaigns |
| Resignation | Hours 48-72 | Collective fatigue, acceptance, "we've survived worse" |

## Follow-Up Scenario Variations

After the initial run, try changing one variable and re-running:

1. **Stage 6 escalation** — "What if Eskom escalates to Stage 6 on day 2?"
2. **Government intervention** — "What if the President addresses the nation on day 1?"
3. **Solar incentive** — "What if government announces emergency solar tax rebate on day 2?"
4. **Competitor enters** — "What if a private power producer announces 500MW available on day 3?"
5. **Protest triggers** — "What if a hospital loses backup power and patients are at risk?"
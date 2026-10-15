"""
Kamptribe Run Contract integration fixture: 24 contracted personas among the full
context graph, the Product Truth, an Assessment Brief whose cohort prose used to be
parsed into false stages, and the committed final Run Contract
"""

import copy
import json
import os
import time

from conftest import make_project

from app.services.local_graph_store import LocalGraphStore
from app.services.simulation_config_generator import SimulationConfigGenerator

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
GRAPH_ID = "graph_kamptribe"
CONTRACT_ID = "kamptribe-run-2c-final"

CONSUMERS = {
    "C01": "C01 - Dutch first-time family", "C02": "C02 - German camper couple",
    "C03": "C03 - Belgian retired couple", "C04": "C04 - UK solo van traveller",
    "C05": "C05 - French young family", "C06": "C06 - Canadian weekend campers",
    "C07": "C07 - US national park first-timers", "C08": "C08 - Swiss motorbike campers",
    "C09": "C09 - Roadtrippers switcher family", "C10": "C10 - RV LIFE switcher couple",
    "C11": "C11 - Google My Maps power planner", "C12": "C12 - Spreadsheet planner",
    "C13": "C13 - ChatGPT itinerary planner", "C14": "C14 - Paper map traditionalist",
    "C15": "C15 - Overlanding couple", "C16": "C16 - Budget student campers",
}
EXISTING = {
    "U01": "U01 - Free tier user", "U02": "U02 - Planner subscriber", "U03": "U03 - Advanced RV user",
    "U04": "U04 - Trip Pass buyer", "U05": "U05 - Lapsed user",
}
ADVOCATES = {"A01": "A01 - Camping blogger", "A02": "A02 - RV YouTuber", "A03": "A03 - Family travel creator"}
CONTEXT = [
    ("Kamptribe", "Product"), ("Google Maps", "Tool"), ("Google My Maps", "Tool"), ("Google Sheets", "Tool"),
    ("ChatGPT", "Tool"), ("Roadtrippers", "Platform"), ("RV LIFE / RV Trip Wizard", "Platform"),
    ("Reddit", "Platform"), ("YouTube", "Platform"), ("Instagram", "Platform"), ("Pinterest", "Platform"),
    ("Threads", "Platform"), ("HERE", "Tool"), ("OpenStreetMap / OSRM", "Tool"), ("Google Places", "Tool"),
    ("National Park Service", "Organization"), ("Parks Canada", "Organization"),
]
PERSONA_NAMES = {**CONSUMERS, **EXISTING, **ADVOCATES}
EXPECTED_IDS = list(CONSUMERS) + list(EXISTING) + list(ADVOCATES)
CONTEXT_NAMES = [n for n, _ in CONTEXT]

# Cohort prose that the brief parser used to turn into private stages
BRIEF = """# Assessment brief - Kamptribe Run 2C

Report for each persona C01-C16:
- organic discovery
- first reaction
- pricing exposure
- outcome

## Cohorts
C03 - Belgian retired couple; opening prompt: they plan every stop in Google My Maps.
C04 - UK solo van traveller; task: compares campground apps before each trip.
C05 - French young family; instruction: tries whatever is free first.

Hypotheses:
H1: Cold prospects discover Kamptribe organically before exposure.
H2: Switchers pay for the Trip Pass rather than a subscription.
"""


def load_contract():
    with open(os.path.join(FIXTURES, "kamptribe_run_contract.json"), encoding="utf-8") as f:
        return json.load(f)


def product_truth():
    with open(os.path.join(FIXTURES, "kamptribe_product_truth.md"), encoding="utf-8") as f:
        return f.read()


def contract_text(contract=None):
    return json.dumps(contract if contract is not None else load_contract(), indent=2)


def build_graph(graph_id=GRAPH_ID, drop=(), extra=()):
    """The context graph: personas (minus drop) + context entities + extra (name, type)"""
    store = LocalGraphStore(graph_id)
    store.set_meta("kamptribe", "", {}, "2026-01-01")
    ids = {}
    for pid, name in PERSONA_NAMES.items():
        if pid in drop:
            continue
        etype = "Consumer" if pid.startswith("C") else "ExistingUser" if pid.startswith("U") else "Advocate"
        ids[name] = store.add_node(name=name, labels=["Entity", etype], summary=f"{name} plans camping trips")
    for name, etype in list(CONTEXT) + list(extra):
        ids[name] = store.add_node(name=name, labels=["Entity", etype], summary=f"{name} (context)")
    return ids


def make_contract_project(contract=None, truth=None, documents=None, contract_name="run_contract.json"):
    """Project with the brief, the Product Truth and the Run Contract (contract=False: none)"""
    docs = [("assessment_brief.md", BRIEF), ("kamptribe_product_truth.md", truth if truth is not None else product_truth()),
            ("camping_market_notes.md", "# Market notes\nFamilies plan routes with Google My Maps and Roadtrippers.")]
    if documents is not None:
        docs = documents
    if contract is not False:
        docs = docs + [(contract_name, contract if isinstance(contract, str) else contract_text(contract))]
    return make_project(requirement=BRIEF, documents=docs)


class FakeConfigLLM:
    """Config generator LLM stand-in: a 40-round time_config and a seed post naming the product"""

    def __init__(self):
        self.prompts = []

    def __call__(self, generator, prompt, system_prompt):
        self.prompts.append(prompt)
        return {
            "total_simulation_hours": 20,
            "minutes_per_round": 30,
            "initial_posts": [
                {"content": "Kamptribe makes camping trips easy, try it now!", "poster_type": "Advocate"},
                {"content": "Camping season is open - where are you heading this summer?", "poster_type": "Advocate"},
            ],
        }


def install_fake_config(monkeypatch):
    fake = FakeConfigLLM()
    monkeypatch.setattr(SimulationConfigGenerator, "_call_llm_with_retry",
                        lambda self, prompt, system_prompt: fake(self, prompt, system_prompt))
    return fake


def create_simulation(client, graph_id=GRAPH_ID):
    response = client.post("/api/simulation/create", json={"project_id": "proj_test", "graph_id": graph_id})
    assert response.status_code == 200, response.json
    return response.json["data"]["simulation_id"]


def setup(client, monkeypatch, contract=None, truth=None, drop=(), extra=(), documents=None, apply=True):
    """Project + graph + simulation (+ applied contract); returns sim_id"""
    install_fake_config(monkeypatch)
    make_contract_project(contract, truth, documents)
    build_graph(drop=drop, extra=extra)
    sim_id = create_simulation(client)
    if apply:
        client.post(f"/api/simulation/{sim_id}/run-contract/apply")
    return sim_id


def preflight(client, sim_id, **params):
    response = client.get(f"/api/simulation/{sim_id}/preflight", query_string=params)
    assert response.status_code == 200, response.json
    return response.json["data"]


def wait_prepare(client, task_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.post("/api/simulation/prepare/status", json={"task_id": task_id}).json["data"]
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(0.05)
    raise AssertionError("prepare timed out")


def prepare(client, sim_id, expect="completed"):
    response = client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "use_llm_for_profiles": False})
    assert response.status_code == 200, response.json
    if response.json["data"].get("already_prepared"):
        return response.json["data"]
    data = wait_prepare(client, response.json["data"]["task_id"])
    assert data["status"] == expect, data
    return data


def mutate(contract, fn):
    data = copy.deepcopy(contract)
    fn(data)
    return data

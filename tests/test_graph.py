"""
Tests use dummy implementations of every Protocol — no real LLM, DB, or network.
"""
from dataclasses import dataclass
from langchain_core.messages import AIMessage
from artcrm_research_agent import create_research_agent, ResearchState


@dataclass(frozen=True)
class DummyMission:
    goal: str = "Find art venues"
    identity: str = "Test Artist"
    targets: str = "galleries, cafes"
    fit_criteria: str = "contemporary art friendly"
    outreach_style: str = "personal"
    language_default: str = "de"


class FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._index = 0

    def invoke(self, messages):
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return AIMessage(content=response)


def make_tools():
    saved = []
    runs = {}

    def web_search(query: str) -> list[dict]:
        return [{"title": f"Result for {query}", "url": "http://example.com", "snippet": "A gallery"}]

    def geo_search(query: str, city: str, country: str = "DE") -> list[dict]:
        return [{"name": "Test Gallery", "address": "Main St 1", "city": city, "country": country}]

    def save_contact(name, city, *, country="DE", type="", website="", email="", phone="", notes="") -> int:
        saved.append({"name": name, "city": city})
        return len(saved)

    def start_run(agent_name: str, input_data: dict) -> int:
        run_id = len(runs) + 1
        runs[run_id] = {"agent": agent_name, "status": "running"}
        return run_id

    def finish_run(run_id: int, status: str, summary: str, output_data: dict) -> None:
        runs[run_id]["status"] = status
        runs[run_id]["summary"] = summary

    return web_search, geo_search, save_contact, start_run, finish_run, saved, runs


def test_agent_saves_contacts():
    web_search, geo_search, save_contact, start_run, finish_run, saved, runs = make_tools()

    llm = FakeLLM([
        '["galleries Munich", "art Munich"]',             # plan_queries response
        '[{"name": "Galerie Nord", "city": "Munich", "country": "DE", "type": "gallery"}]',  # extract_contacts
    ])

    agent = create_research_agent(
        llm=llm,
        web_search=web_search,
        geo_search=geo_search,
        save_contact=save_contact,
        start_run=start_run,
        finish_run=finish_run,
        mission=DummyMission(),
    )

    result = agent.invoke({"city": "Munich", "industry": "gallery", "country": "DE"})

    assert len(result["saved_ids"]) == 1
    assert saved[0]["name"] == "Galerie Nord"
    assert "1 new contacts saved" in result["summary"]
    assert result["errors"] == []


def test_agent_handles_empty_search_results():
    web_search, geo_search, save_contact, start_run, finish_run, saved, runs = make_tools()

    def empty_geo_search(query, city, country="DE"):
        return []

    def empty_web_search(query):
        return []

    llm = FakeLLM([
        '["galleries Munich"]',  # plan_queries
        '[]',                    # extract_contacts — nothing found
    ])

    agent = create_research_agent(
        llm=llm,
        web_search=empty_web_search,
        geo_search=empty_geo_search,
        save_contact=save_contact,
        start_run=start_run,
        finish_run=finish_run,
        mission=DummyMission(),
    )

    result = agent.invoke({"city": "Munich", "industry": "gallery"})

    assert result["saved_ids"] == []
    assert result["errors"] == []
    assert "0 new contacts saved" in result["summary"]


def test_agent_handles_llm_json_error():
    web_search, geo_search, save_contact, start_run, finish_run, saved, runs = make_tools()

    llm = FakeLLM(["this is not json"])  # will fail parse in plan_queries

    agent = create_research_agent(
        llm=llm,
        web_search=web_search,
        geo_search=geo_search,
        save_contact=save_contact,
        start_run=start_run,
        finish_run=finish_run,
        mission=DummyMission(),
    )

    result = agent.invoke({"city": "Munich", "industry": "gallery"})

    assert len(result["errors"]) > 0
    assert "plan_queries" in result["errors"][0]
    assert result["saved_ids"] == []


def test_agent_handles_markdown_wrapped_json():
    web_search, geo_search, save_contact, start_run, finish_run, saved, runs = make_tools()

    llm = FakeLLM([
        '```json\n["galleries Munich"]\n```',
        '```json\n[{"name": "Galerie Süd", "city": "Munich"}]\n```',
    ])

    agent = create_research_agent(
        llm=llm,
        web_search=web_search,
        geo_search=geo_search,
        save_contact=save_contact,
        start_run=start_run,
        finish_run=finish_run,
        mission=DummyMission(),
    )

    result = agent.invoke({"city": "Munich", "industry": "gallery"})

    assert len(result["saved_ids"]) == 1
    assert result["errors"] == []

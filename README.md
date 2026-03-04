# artcrm-research-agent

LangGraph agent that researches cities and industries for potential contacts. Saves results to the database as `status=candidate` for the scout agent to score.

## What it does

Given a city and industry (e.g. `Munich` / `gallery`), it:
1. Plans a set of search queries using the LLM
2. Runs geo search (OpenStreetMap Overpass) and web search
3. Extracts structured contact info from results with the LLM
4. Saves new contacts to the database (deduplication by name+city)

## Usage

```python
from artcrm_research_agent import create_research_agent

agent = create_research_agent(
    llm=your_llm,
    web_search=your_web_search_fn,
    geo_search=your_geo_search_fn,
    save_contact=your_save_contact_fn,
    start_run=your_start_run_fn,
    finish_run=your_finish_run_fn,
    mission=your_mission,
)

result = agent.invoke({"city": "Munich", "industry": "gallery"})
print(result["summary"])
# "research_agent: Munich/gallery — 12 new contacts saved"
```

## Protocols

All dependencies are injected. Each callable must match the Protocol defined in [protocols.py](artcrm_research_agent/protocols.py):

| Parameter | Protocol | Description |
|---|---|---|
| `llm` | `LanguageModel` | Any LangChain `BaseChatModel` |
| `web_search` | `WebSearcher` | `(query: str) -> list[dict]` |
| `geo_search` | `GeoSearcher` | `(query, city, country) -> list[dict]` |
| `save_contact` | `ContactSaver` | `(name, city, **kwargs) -> int` |
| `start_run` | `RunStarter` | `(agent_name, input_data) -> int` |
| `finish_run` | `RunFinisher` | `(run_id, status, summary, output_data) -> None` |
| `mission` | `AgentMission` | Any object with `goal`, `identity`, `targets`, `fit_criteria`, `outreach_style`, `language_default` |

## Reconfiguring for a different domain

Pass a different `mission` object at instantiation. The graph, logic, and tools are unchanged.

## Testing

```bash
uv run pytest -v
```

Tests use dummy tool implementations — no real LLM, DB, or network required.

## Support

If you find this useful, a small donation helps keep projects like this going:
[Donate via PayPal](https://paypal.me/christopherrehm001)

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from .protocols import AgentMission, LanguageModel, WebSearcher, GeoSearcher, PageFetcher, ContactSaver, RunStarter, RunFinisher
from .state import ResearchState
from .prompts import extract_contacts_prompt
from ._utils import parse_json_response

# Fixed Google Maps search terms per scan level
LEVEL_TERMS: dict[int, list[str]] = {
    1: ["Kunstgalerie", "Galerie", "Café", "Kaffeehaus", "Innenarchitekt", "Raumausstatter", "Coworking Space"],
    2: ["Geschenkeladen", "Esoterikladen", "Kristallladen", "Yoga Studio", "Concept Store", "Designladen", "Boutique"],
    3: ["Restaurant", "Gasthaus", "Bistro", "Weinrestaurant", "Gasthof"],
    4: ["Firmensitz", "Hauptverwaltung", "Bürogebäude", "Unternehmensberatung", "Technologieunternehmen"],
    5: ["Hotel", "Boutique Hotel", "Design Hotel", "Landhotel", "Stadthotel"],
}


def create_research_agent(
    llm: LanguageModel,
    web_search: WebSearcher,
    geo_search: GeoSearcher,
    fetch_page: PageFetcher,
    save_contact: ContactSaver,
    start_run: RunStarter,
    finish_run: RunFinisher,
    mission: AgentMission,
):
    """
    Build and return a compiled LangGraph research agent.

    Scans a city at a given level (1-5). Uses Google Maps for structured venue
    discovery and DuckDuckGo + page fetching for contact detail extraction.

    Usage:
        agent = create_research_agent(llm=..., ...)
        result = agent.invoke({"city": "Konstanz", "country": "DE", "level": 1})
        print(result["summary"])
    """

    def init(state: ResearchState) -> dict:
        level = state.get("level", 1)
        run_id = start_run("research_agent", {
            "city": state["city"],
            "country": state.get("country", "DE"),
            "level": level,
        })
        return {
            "run_id": run_id,
            "country": state.get("country", "DE"),
            "level": level,
            "maps_terms": LEVEL_TERMS.get(level, LEVEL_TERMS[1]),
            "raw_results": [],
            "contacts_to_save": [],
            "saved_ids": [],
            "errors": [],
            "summary": "",
        }

    def run_maps_search(state: ResearchState) -> dict:
        """Run each Google Maps term for this level. Collects structured venue data."""
        results = []
        for term in state.get("maps_terms", []):
            try:
                hits = geo_search(term, state["city"], state.get("country", "DE"))
                results.extend(hits)
            except Exception as e:
                pass  # individual term failures are non-fatal
        # Deduplicate by name (case-insensitive)
        seen = set()
        deduped = []
        for r in results:
            key = r.get("name", "").lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(r)
        return {"raw_results": deduped}

    def run_web_search(state: ResearchState) -> dict:
        """Run 2 targeted web searches per level to supplement Maps data."""
        level = state.get("level", 1)
        city = state["city"]
        level_labels = {
            1: f"Kunstgalerie Innenarchitekt Coworking {city}",
            2: f"Concept Store Esoterikladen Boutique {city}",
            3: f"Restaurant Gasthaus {city} Empfehlung",
            4: f"Firmensitz Unternehmen Hauptverwaltung {city}",
            5: f"Hotel Boutique Hotel {city}",
        }
        queries = [level_labels.get(level, f"{city} venues")]
        extra = {
            1: f"Galerie {city} zeitgenössische Kunst",
            2: f"Geschenke Wellness Shop {city}",
            3: f"bestes Restaurant {city}",
            4: f"größte Unternehmen {city} Kunst Büro",
            5: f"Design Hotel {city} Boutique",
        }
        queries.append(extra.get(level, f"{city} art venues"))

        web_results = list(state.get("raw_results", []))
        for query in queries:
            try:
                web_results.extend(web_search(query))
            except Exception:
                pass
        return {"raw_results": web_results}

    def fetch_pages(state: ResearchState) -> dict:
        """Fetch top web result pages to get full contact details beyond snippets."""
        if not state.get("raw_results"):
            return {}
        seen = set()
        urls = []
        for r in state["raw_results"]:
            url = r.get("url", "") or r.get("website", "")
            if url and url not in seen and not url.startswith("https://www.google"):
                seen.add(url)
                urls.append(url)
        pages = []
        for url in urls[:3]:
            content = fetch_page(url)
            if content:
                pages.append({"source": "page", "url": url, "content": content[:1500]})
        if pages:
            return {"raw_results": state["raw_results"] + pages}
        return {}

    def extract_contacts(state: ResearchState) -> dict:
        if not state.get("raw_results"):
            return {"contacts_to_save": []}
        level = state.get("level", 1)
        system, user = extract_contacts_prompt(mission, state["city"], level, state["raw_results"])
        try:
            response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            contacts = parse_json_response(response.content)
            if not isinstance(contacts, list):
                raise ValueError("Expected a JSON array")
        except Exception as e:
            return {"errors": state.get("errors", []) + [f"extract_contacts: {e}"], "contacts_to_save": []}
        return {"contacts_to_save": contacts}

    def save_contacts(state: ResearchState) -> dict:
        saved_ids = []
        for contact in state.get("contacts_to_save", []):
            try:
                contact_id = save_contact(
                    name=contact.get("name", ""),
                    city=contact.get("city", state["city"]),
                    country=contact.get("country", state.get("country", "DE")),
                    type=contact.get("type", ""),
                    website=contact.get("website", ""),
                    email=contact.get("email", ""),
                    phone=contact.get("phone", ""),
                    notes=contact.get("notes", ""),
                )
                if contact_id:
                    saved_ids.append(contact_id)
            except Exception:
                pass
        return {"saved_ids": saved_ids}

    def generate_report(state: ResearchState) -> dict:
        n = len(state.get("saved_ids", []))
        errs = state.get("errors", [])
        city = state["city"]
        level = state.get("level", 1)
        if errs:
            summary = f"research_agent: {city} level {level} — {n} contacts saved, {len(errs)} error(s): {errs[0]}"
            status = "failed" if n == 0 else "completed"
        else:
            summary = f"research_agent: {city} level {level} — {n} new contacts saved"
            status = "completed"
        finish_run(
            state.get("run_id", 0),
            status,
            summary,
            {"saved_count": n, "level": level, "errors": errs},
        )
        return {"summary": summary}

    graph = StateGraph(ResearchState)
    graph.add_node("init", init)
    graph.add_node("run_maps_search", run_maps_search)
    graph.add_node("run_web_search", run_web_search)
    graph.add_node("fetch_pages", fetch_pages)
    graph.add_node("extract_contacts", extract_contacts)
    graph.add_node("save_contacts", save_contacts)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("init")
    graph.add_edge("init", "run_maps_search")
    graph.add_edge("run_maps_search", "run_web_search")
    graph.add_edge("run_web_search", "fetch_pages")
    graph.add_edge("fetch_pages", "extract_contacts")
    graph.add_edge("extract_contacts", "save_contacts")
    graph.add_edge("save_contacts", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()

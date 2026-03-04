from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from .protocols import AgentMission, LanguageModel, WebSearcher, GeoSearcher, ContactSaver, RunStarter, RunFinisher
from .state import ResearchState
from .prompts import plan_queries_prompt, extract_contacts_prompt
from ._utils import parse_json_response


def create_research_agent(
    llm: LanguageModel,
    web_search: WebSearcher,
    geo_search: GeoSearcher,
    save_contact: ContactSaver,
    start_run: RunStarter,
    finish_run: RunFinisher,
    mission: AgentMission,
):
    """
    Build and return a compiled LangGraph research agent.

    The agent researches a city + industry for potential contacts and saves them
    to the database with status='candidate'.

    Usage:
        agent = create_research_agent(llm=..., web_search=..., ...)
        result = agent.invoke({"city": "Munich", "industry": "gallery"})
        print(result["summary"])

    Tool interfaces are defined in protocols.py. Pass any callable that matches
    the Protocol signature — no imports from this package required.
    """

    def init(state: ResearchState) -> dict:
        run_id = start_run("research_agent", {"city": state["city"], "industry": state["industry"]})
        return {
            "run_id": run_id,
            "country": state.get("country", "DE"),
            "search_queries": [],
            "raw_results": [],
            "contacts_to_save": [],
            "saved_ids": [],
            "errors": [],
            "summary": "",
        }

    def plan_queries(state: ResearchState) -> dict:
        system, user = plan_queries_prompt(mission, state["city"], state["industry"], state["country"])
        try:
            response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            queries = parse_json_response(response.content)
            if not isinstance(queries, list):
                raise ValueError("Expected a JSON array")
        except Exception as e:
            return {"errors": state["errors"] + [f"plan_queries: {e}"]}
        return {"search_queries": queries}

    def run_searches(state: ResearchState) -> dict:
        if state.get("errors"):
            return {}
        results = []
        for query in state["search_queries"]:
            try:
                results.extend(geo_search(query, state["city"], state["country"]))
            except Exception as e:
                return {"errors": state["errors"] + [f"geo_search '{query}': {e}"]}
        # supplement with web search for the first two queries
        for query in state["search_queries"][:2]:
            try:
                results.extend(web_search(query))
            except Exception as e:
                # web search failure is non-fatal
                pass
        return {"raw_results": results}

    def extract_contacts(state: ResearchState) -> dict:
        if state.get("errors") or not state.get("raw_results"):
            return {"contacts_to_save": []}
        system, user = extract_contacts_prompt(mission, state["city"], state["industry"], state["raw_results"])
        try:
            response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            contacts = parse_json_response(response.content)
            if not isinstance(contacts, list):
                raise ValueError("Expected a JSON array")
        except Exception as e:
            return {"errors": state["errors"] + [f"extract_contacts: {e}"], "contacts_to_save": []}
        return {"contacts_to_save": contacts}

    def save_contacts(state: ResearchState) -> dict:
        saved_ids = []
        for contact in state.get("contacts_to_save", []):
            try:
                contact_id = save_contact(
                    name=contact.get("name", ""),
                    city=contact.get("city", state["city"]),
                    country=contact.get("country", state["country"]),
                    type=contact.get("type", ""),
                    website=contact.get("website", ""),
                    email=contact.get("email", ""),
                    phone=contact.get("phone", ""),
                    notes=contact.get("notes", ""),
                )
                if contact_id:
                    saved_ids.append(contact_id)
            except Exception as e:
                pass  # individual save failures don't stop the run
        return {"saved_ids": saved_ids}

    def generate_report(state: ResearchState) -> dict:
        n = len(state.get("saved_ids", []))
        errs = state.get("errors", [])
        city = state["city"]
        industry = state["industry"]
        if errs:
            summary = f"research_agent: {city}/{industry} — {n} contacts saved, {len(errs)} error(s): {errs[0]}"
            status = "failed" if n == 0 else "completed"
        else:
            summary = f"research_agent: {city}/{industry} — {n} new contacts saved"
            status = "completed"
        finish_run(
            state.get("run_id", 0),
            status,
            summary,
            {"saved_count": n, "errors": errs},
        )
        return {"summary": summary}

    graph = StateGraph(ResearchState)
    graph.add_node("init", init)
    graph.add_node("plan_queries", plan_queries)
    graph.add_node("run_searches", run_searches)
    graph.add_node("extract_contacts", extract_contacts)
    graph.add_node("save_contacts", save_contacts)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("init")
    graph.add_edge("init", "plan_queries")
    graph.add_edge("plan_queries", "run_searches")
    graph.add_edge("run_searches", "extract_contacts")
    graph.add_edge("extract_contacts", "save_contacts")
    graph.add_edge("save_contacts", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()

from typing import TypedDict


class ResearchState(TypedDict):
    # --- inputs ---
    city: str
    industry: str
    country: str        # ISO 3166-1 alpha-2, default "DE"

    # --- working state ---
    run_id: int
    search_queries: list[str]
    raw_results: list[dict]
    contacts_to_save: list[dict]
    saved_ids: list[int]
    errors: list[str]

    # --- output ---
    summary: str

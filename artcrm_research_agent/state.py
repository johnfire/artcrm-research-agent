from typing import TypedDict


class ResearchState(TypedDict):
    # --- inputs ---
    city: str
    country: str        # ISO 3166-1 alpha-2, default "DE"
    level: int          # scan level 1-5

    # --- working state ---
    run_id: int
    maps_terms: list[str]       # fixed terms for this level
    raw_results: list[dict]     # from Google Maps + web search + fetched pages
    contacts_to_save: list[dict]
    saved_ids: list[int]
    errors: list[str]

    # --- output ---
    summary: str

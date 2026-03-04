from .protocols import AgentMission


def plan_queries_prompt(mission: AgentMission, city: str, industry: str, country: str) -> tuple[str, str]:
    system = (
        f"You are a research assistant helping {mission.identity} find potential clients.\n"
        f"Mission: {mission.goal}\n"
        f"Looking for: {mission.targets}"
    )
    user = (
        f"Generate 3-5 specific search queries to find {industry} venues in {city}, {country}.\n"
        f"Focus on places likely to match: {mission.fit_criteria}\n\n"
        f"Return a JSON array of search query strings only.\n"
        f'Example: ["galleries Munich", "contemporary art Munich gallery"]\n'
        f"Return ONLY the JSON array, no other text."
    )
    return system, user


def extract_contacts_prompt(
    mission: AgentMission,
    city: str,
    industry: str,
    raw_results: list[dict],
) -> tuple[str, str]:
    import json
    system = (
        f"You are extracting contact information for {mission.identity}.\n"
        f"Mission: {mission.goal}\n"
        f"Target venues: {mission.targets}"
    )
    user = (
        f"From these search results for {industry} venues in {city}, extract contact information.\n"
        f"Only include venues that could match: {mission.fit_criteria}\n\n"
        f"For each venue, extract:\n"
        f"- name (required)\n"
        f"- city (required)\n"
        f"- country (2-letter ISO code)\n"
        f"- type (gallery/restaurant/hotel/office/cafe/museum/other)\n"
        f"- address\n"
        f"- website\n"
        f"- email\n"
        f"- phone\n"
        f"- notes (one sentence: why this could be a good fit)\n\n"
        f"Search results:\n{json.dumps(raw_results, ensure_ascii=False, indent=2)}\n\n"
        f"Return a JSON array of objects. If nothing relevant found, return [].\n"
        f"Return ONLY the JSON array, no other text."
    )
    return system, user

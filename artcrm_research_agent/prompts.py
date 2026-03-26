from .protocols import AgentMission

LEVEL_DESCRIPTIONS = {
    1: "galleries, cafes, interior designers, and coworking spaces",
    2: "gift shops, esoteric/wellness shops, and concept stores",
    3: "independent restaurants",
    4: "corporate offices and headquarters",
    5: "hotels",
}


def extract_contacts_prompt(
    mission: AgentMission,
    city: str,
    level: int,
    raw_results: list[dict],
) -> tuple[str, str]:
    import json
    level_desc = LEVEL_DESCRIPTIONS.get(level, "venues")
    system = (
        f"You are extracting contact information for {mission.identity}.\n"
        f"Mission: {mission.goal}\n"
        f"You are scanning for: {level_desc}"
    )
    user = (
        f"From these search results for {level_desc} in {city}, extract every venue found.\n\n"
        f"Fit criteria for later scoring — use this to write useful notes:\n{mission.fit_criteria}\n\n"
        f"For EVERY venue found, extract:\n"
        f"- name (required)\n"
        f"- city (default: {city})\n"
        f"- country (2-letter ISO code)\n"
        f"- type (gallery/restaurant/hotel/cafe/interior_designer/coworking/corporate_office/concept_store/gift_shop/wellness/other)\n"
        f"- address\n"
        f"- website\n"
        f"- email\n"
        f"- phone\n"
        f"- notes: 2-3 sentences:\n"
        f"  1. What the venue is and does\n"
        f"  2. Signals about artist level, style, or openness (e.g. 'shows emerging regional artists', 'only blue-chip names', 'design-conscious interior')\n"
        f"  3. Fit assessment: strong fit / weak fit / unclear — be specific\n\n"
        f"Include ALL venues from the results — do not filter here. The scout agent will score and drop bad fits.\n"
        f"If a venue clearly only shows internationally established artists, still include it — note it in the notes field.\n\n"
        f"Search results:\n{json.dumps(raw_results, ensure_ascii=False, indent=2)[:7000]}\n\n"
        f"Return a JSON array of objects. If nothing found at all, return [].\n"
        f"Return ONLY the JSON array, no other text."
    )
    return system, user

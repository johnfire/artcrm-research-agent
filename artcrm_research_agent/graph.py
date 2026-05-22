"""
Research agent.

Scans a city at a given level (1-5) for potential contacts. Uses Google Maps
for structured venue discovery and Brave Search + page fetching for additional
venues and contact details.

Pipeline position: research → enrich → scout → outreach → followup
"""
import json
import logging
import re

from langchain_core.messages import SystemMessage, HumanMessage

from .protocols import AgentMission, LanguageModel, WebSearcher, GeoSearcher, PageFetcher, ContactSaver, RunStarter, RunFinisher, ChainsFetcher
from .prompts import extract_contacts_prompt
from ._utils import parse_json_response

logger = logging.getLogger(__name__)

# Fixed Google Maps search terms per scan level
LEVEL_TERMS: dict[int, list[str]] = {
    1: ["Kunstgalerie", "Galerie", "Café", "Kaffeehaus", "Innenarchitekt", "Raumausstatter", "Coworking Space"],
    2: ["Geschenkeladen", "Esoterikladen", "Kristallladen", "Yoga Studio", "Concept Store", "Designladen", "Boutique"],
    3: ["Restaurant", "Gasthaus", "Bistro", "Weinrestaurant", "Gasthof"],
    4: ["Firmensitz", "Hauptverwaltung", "Bürogebäude", "Unternehmensberatung", "Technologieunternehmen"],
    5: ["Hotel", "Boutique Hotel", "Design Hotel", "Landhotel", "Stadthotel"],
}

# Web search queries per level (primary + supplemental)
_WEB_QUERIES: dict[int, tuple[str, str]] = {
    1: ("Kunstgalerie Innenarchitekt Coworking {city}", "Galerie {city} zeitgenössische Kunst"),
    2: ("Concept Store Esoterikladen Boutique {city}", "Geschenke Wellness Shop {city}"),
    3: ("Restaurant Gasthaus {city} Empfehlung", "bestes Restaurant {city}"),
    4: ("Firmensitz Unternehmen Hauptverwaltung {city}", "größte Unternehmen {city} Kunst Büro"),
    5: ("Hotel Boutique Hotel {city}", "Design Hotel {city} Boutique"),
}

def _normalize_chain(name: str) -> str:
    return re.sub(r"[\s\-_/&'\".,;:!?]+", " ", name.lower()).strip()


def _is_ignored_chain(name: str, chains: list[str], threshold: float = 0.90) -> bool:
    import difflib
    n = _normalize_chain(name)
    for chain in chains:
        c = _normalize_chain(chain)
        if n == c or n.startswith(c + " "):
            return True
        sm = difflib.SequenceMatcher(None, n, c)
        if sm.quick_ratio() >= threshold and sm.ratio() >= threshold:
            return True
    return False


# Skip fetching directory/social sites — they won't have venue-specific contact details
_SKIP_FETCH_DOMAINS = re.compile(
    r"(google\.|facebook\.|instagram\.|yelp\.|tripadvisor\.|gelbeseiten\.|"
    r"yellowpages\.|booking\.|maps\.|wikipedia\.|openstreetmap\.)",
    re.IGNORECASE,
)


class _ResearchAgent:
    def __init__(
        self,
        llm: LanguageModel,
        web_search: WebSearcher,
        geo_search: GeoSearcher,
        fetch_page: PageFetcher,
        save_contact: ContactSaver,
        start_run: RunStarter,
        finish_run: RunFinisher,
        mission: AgentMission,
        fetch_chains: ChainsFetcher | None = None,
    ):
        self._llm = llm
        self._web_search = web_search
        self._geo_search = geo_search
        self._fetch_page = fetch_page
        self._save_contact = save_contact
        self._start_run = start_run
        self._finish_run = finish_run
        self._mission = mission
        self._fetch_chains = fetch_chains

    def invoke(self, inputs: dict) -> dict:
        city = inputs["city"]
        country = inputs.get("country", "DE")
        level = inputs.get("level", 1)

        run_id = self._start_run("research_agent", {"city": city, "country": country, "level": level})
        errors = []

        ignored_chains = self._fetch_chains() if self._fetch_chains else []

        maps_results = self._run_maps_search(city, country, level)
        web_results = self._run_web_search(city, level)
        page_texts = self._fetch_pages(web_results)

        contacts, err = self._extract_contacts(city, level, maps_results, web_results, page_texts)
        if err:
            errors.append(err)

        if ignored_chains:
            before = len(contacts)
            contacts = [c for c in contacts if not _is_ignored_chain(c.get("name", ""), ignored_chains)]
            skipped = before - len(contacts)
            if skipped:
                logger.info("research: skipped %d ignored chain(s) in %s", skipped, city)

        contacts = self._fetch_missing_emails(contacts, city)

        saved_ids = self._save_contacts(contacts, city, country, level)

        n = len(saved_ids)
        if errors:
            summary = f"research_agent: {city} level {level} — {n} contacts saved, {len(errors)} error(s): {errors[0]}"
            status = "failed" if n == 0 else "completed"
        else:
            summary = f"research_agent: {city} level {level} — {n} new contacts saved"
            status = "completed"

        self._finish_run(run_id, status, summary, {"saved_count": n, "level": level, "errors": errors})
        logger.info(summary)
        return {"summary": summary, "saved_ids": saved_ids}

    def _run_maps_search(self, city: str, country: str, level: int) -> list[dict]:
        """Run each Maps term for this level. Deduplicates by name."""
        terms = LEVEL_TERMS.get(level, LEVEL_TERMS[1])
        results = []
        for term in terms:
            try:
                hits = self._geo_search(term, city, country)
                results.extend(hits)
            except Exception:
                pass

        seen: set[str] = set()
        deduped = []
        for r in results:
            key = r.get("name", "").lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(r)
        logger.info("research: %s Maps hits for %s level %d", len(deduped), city, level)
        return deduped

    def _run_web_search(self, city: str, level: int) -> list[dict]:
        """Run 2 targeted web searches for this level."""
        primary, secondary = _WEB_QUERIES.get(level, (f"{city} venues", f"{city} art venues"))
        results = []
        for query in [primary.format(city=city), secondary.format(city=city)]:
            try:
                results.extend(self._web_search(query=query))
            except Exception:
                pass
        return results

    def _fetch_pages(self, web_results: list[dict]) -> list[str]:
        """Fetch up to 3 web result pages for fuller contact details."""
        seen: set[str] = set()
        page_texts = []
        for r in web_results:
            url = r.get("url", "")
            if not url or url in seen or _SKIP_FETCH_DOMAINS.search(url):
                continue
            seen.add(url)
            try:
                text = self._fetch_page(url)
                if text:
                    page_texts.append(f"[Page: {url}]\n{text[:1500]}")
            except Exception:
                pass
            if len(page_texts) >= 3:
                break
        return page_texts

    def _extract_contacts(
        self,
        city: str,
        level: int,
        maps_results: list[dict],
        web_results: list[dict],
        page_texts: list[str],
    ) -> tuple[list[dict], str | None]:
        """Ask the LLM to extract contacts from all collected data."""
        if not maps_results and not web_results:
            return [], None
        system, user = extract_contacts_prompt(
            self._mission, city, level, maps_results, web_results, page_texts
        )
        try:
            response = self._llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            contacts = parse_json_response(response.content)
            if not isinstance(contacts, list):
                raise ValueError("Expected a JSON array")
            return contacts, None
        except Exception as e:
            return [], f"extract_contacts: {e}"

    def _fetch_missing_emails(self, contacts: list[dict], city: str) -> list[dict]:
        """For each contact with a website but no email, fetch the page and extract an email.
        Contacts with no website AND no email after fetching are flagged _no_data=True."""
        import re
        email_re = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
        noise_domains = {
            "example.com", "sentry.io", "wixpress.com", "squarespace.com",
            "wordpress.com", "shopify.com", "amazonaws.com", "googletagmanager.com",
        }
        found = 0
        no_data = 0
        for contact in contacts:
            if contact.get("email"):
                continue
            if not contact.get("website"):
                contact["_no_data"] = True
                no_data += 1
                continue
            try:
                text = self._fetch_page(contact["website"])
                if not text:
                    contact["_no_data"] = True
                    no_data += 1
                    continue
                for match in email_re.finditer(text):
                    email = match.group(0).lower()
                    domain = email.split("@")[1]
                    if domain not in noise_domains:
                        contact["email"] = email
                        found += 1
                        logger.info("research: email found for %s — %s", contact.get("name", ""), email)
                        break
                else:
                    contact["_no_data"] = True
                    no_data += 1
            except Exception:
                contact["_no_data"] = True
                no_data += 1
        if found:
            logger.info("research: fetched emails for %d contact(s) in %s", found, city)
        if no_data:
            logger.info("research: %d contact(s) have no web presence in %s — will save as cannot_find_more_data", no_data, city)
        return contacts

    def _save_contacts(self, contacts: list[dict], city: str, country: str, level: int) -> list[int]:
        saved_ids = []
        for contact in contacts:
            try:
                status = "cannot_find_more_data" if contact.get("_no_data") else "candidate"
                contact_id = self._save_contact(
                    name=contact.get("name", ""),
                    city=contact.get("city", city),
                    country=contact.get("country", country),
                    type=contact.get("type", ""),
                    website=contact.get("website", ""),
                    email=contact.get("email", ""),
                    phone=contact.get("phone", ""),
                    notes=contact.get("notes", ""),
                    scan_level=level,
                    status=status,
                )
                if contact_id:
                    saved_ids.append(contact_id)
            except Exception:
                pass
        return saved_ids


def create_research_agent(
    llm: LanguageModel,
    web_search: WebSearcher,
    geo_search: GeoSearcher,
    fetch_page: PageFetcher,
    save_contact: ContactSaver,
    start_run: RunStarter,
    finish_run: RunFinisher,
    mission: AgentMission,
    fetch_chains: ChainsFetcher | None = None,
) -> _ResearchAgent:
    """
    Build and return a research agent.

    Scans a city at a given level (1-5). Uses Google Maps for structured venue
    discovery and Brave Search + page fetching for additional contacts.

    Usage:
        agent = create_research_agent(llm=..., ...)
        result = agent.invoke({"city": "Konstanz", "country": "DE", "level": 1})
        print(result["summary"])
    """
    return _ResearchAgent(
        llm=llm,
        web_search=web_search,
        geo_search=geo_search,
        fetch_page=fetch_page,
        save_contact=save_contact,
        start_run=start_run,
        finish_run=finish_run,
        mission=mission,
        fetch_chains=fetch_chains,
    )

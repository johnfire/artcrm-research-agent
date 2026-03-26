from typing import Any, Protocol


class AgentMission(Protocol):
    """
    What the agent system is working toward.
    Any object with these string attributes satisfies this protocol.
    The supervisor's Mission dataclass works here without any changes.
    """
    goal: str
    identity: str
    targets: str
    fit_criteria: str
    outreach_style: str
    language_default: str


class LanguageModel(Protocol):
    """Any LangChain-compatible chat model (BaseChatModel) satisfies this."""
    def invoke(self, messages: list) -> Any:
        """Returns an object with a .content (str) attribute."""
        ...


class WebSearcher(Protocol):
    """Search the web. Returns list of results with title, url, snippet."""
    def __call__(self, query: str) -> list[dict]: ...


class PageFetcher(Protocol):
    """Fetch a URL and return its plain text content. Returns empty string on failure."""
    def __call__(self, url: str) -> str: ...


class GeoSearcher(Protocol):
    """Search for venues by location. Returns list of results with name, address, city."""
    def __call__(self, query: str, city: str, country: str = "DE") -> list[dict]: ...


class ContactSaver(Protocol):
    """Save a contact to the database. Returns contact_id, or 0 on duplicate."""
    def __call__(
        self,
        name: str,
        city: str,
        *,
        country: str = "DE",
        type: str = "",
        website: str = "",
        email: str = "",
        phone: str = "",
        notes: str = "",
    ) -> int: ...


class RunStarter(Protocol):
    """Log the start of an agent run. Returns run_id."""
    def __call__(self, agent_name: str, input_data: dict) -> int: ...


class RunFinisher(Protocol):
    """Log the completion of an agent run."""
    def __call__(self, run_id: int, status: str, summary: str, output_data: dict) -> None: ...

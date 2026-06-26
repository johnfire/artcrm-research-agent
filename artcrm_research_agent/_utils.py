import json
import re


def parse_json_response(text: str):
    """Parse JSON (object OR array) from an LLM response.

    Tolerates markdown code fences and surrounding prose. Tries a direct parse
    first, then falls back to extracting the first balanced JSON array or object
    embedded in the text. Raises json.JSONDecodeError if nothing parses.

    Shared, identical implementation across all agent packages (see review M-6):
    keep these byte-for-byte in sync until they are factored into a common
    package (M-7).
    """
    cleaned = text.strip()
    # Strip a leading ```json / ``` fence and a trailing ``` fence.
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to the first JSON array or object found in surrounding prose.
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise

import json
import re


def parse_json_response(text: str):
    """
    Parse JSON from an LLM response.
    Handles markdown code fences that some models wrap output in.
    Raises json.JSONDecodeError if parsing fails.
    """
    text = text.strip()
    # strip ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())

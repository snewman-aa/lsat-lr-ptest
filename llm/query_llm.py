import os
import json
from pathlib import Path
from typing import Any

from clients.gemini_client import GeminiClient

from loguru import logger


from config import load_config
_config = load_config(Path(__file__).parent.parent / "config.yaml")
_llm_cfg = _config.get("llm", {})
_provider = _llm_cfg.get("provider", "gemini").lower()
_model = _llm_cfg.get("model")
_api_key_env = _llm_cfg.get("api_key_env")

if _provider == "gemini":
    if GeminiClient is None:
        raise ImportError("clients.gemini_client.GeminiClient required for provider 'gemini'")
else:
    raise ValueError(f"Unsupported LLM provider '{_provider}'")


_SYSTEM_PROMPT = (
    "You are an expert LSAT question writer."
    " Generate a single official-style LSAT question that fits the user's request."
    " Respond strictly in JSON with keys: 'stimulus', 'prompt', 'explanation'."
    " The 'stimulus' should be a short paragraph, the 'prompt' should be a question that presents how the"
    " test-taker should reason over the stimulus. The 'explanation' should be a short paragraph"
    " explaining the correct answer."
)


def _build_messages(user_request: str) -> list[dict[str, str]]:
    """
    Construct the message list for chat-based LLMs.
    """
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_request}
    ]


def generate_sample_question(user_prompt: str) -> dict[str, Any]:
    """
    Generate an LSAT question via the configured LLM.

    :param user_prompt: The user’s specification
    :return: Parsed JSON with keys 'stimulus', 'prompt', 'answers', 'correct_answer', 'explanation'
    """
    messages = _build_messages(user_prompt)

    if _provider == "gemini":
        gemini = GeminiClient(api_key=os.getenv(_api_key_env), model_name=_model)
        content = gemini.chat(messages)
    else:
        raise RuntimeError(f"Unsupported LLM provider '{_provider}'")

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # attempt to extract the JSON portion of the response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            snippet = content[start:end]
            try:
                data = json.loads(snippet)
            except Exception:
                raise ValueError(f"Failed to parse JSON snippet: {snippet}")
        else:
            raise ValueError(f"No JSON found in LLM output: {content}")

    # validate keys
    required = {"stimulus", "prompt", "explanation"}
    if not required.issubset(data.keys()):
        missing = required - set(data.keys())
        raise KeyError(f"Missing keys in LLM output: {missing}")

    logger.debug(data)

    return data

if __name__ == '__main__':
    q = generate_sample_question('Generate a question that weakens an argument on climate change.')
    print(q)

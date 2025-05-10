import os
from pydantic import ValidationError
from google import genai
from google.genai import types
from loguru import logger

from lsat_lr_ptest.config import load_config
from lsat_lr_ptest.app.models import SampleQuestion


class GeminiClient:
    def __init__(self):
        cfg = load_config()
        api_key = os.getenv(cfg.llm.api_key_env)
        if not api_key:
            raise ValueError(f"Set ${cfg.llm.api_key_env} in your environment")

        self.client = genai.Client(api_key=api_key)
        self.model_name = cfg.llm.model
        self.system_instruction = (
            "You are an expert LSAT question writer."
            " Generate a single official-style LSAT question that fits the user's request."
            " Respond strictly in JSON with keys: 'stimulus', 'prompt', 'explanation'."
            " The 'stimulus' should be a short paragraph, the 'prompt' should be a"
            " question that presents how the test-taker should reason over the stimulus."
            " The 'explanation' should be a short paragraph explaining the correct answer."
        )

    def generate_sample_question(self, user_request: str) -> SampleQuestion:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_request,
            config=types.GenerateContentConfig(
                system_instruction  = self.system_instruction,
                response_mime_type  = "application/json",
                response_schema     = SampleQuestion,
            ),
        )

        try:
            sample: SampleQuestion = response.parsed
        except ValidationError as e:
            raise ValueError(f"Invalid JSON from Gemini:\n{e}\nRaw text:\n{response.text}")

        logger.debug(sample.model_dump_json())
        return sample

from google import genai
from google.genai import types


class GeminiClient:
    """
    Google Gemini 2.0 Flash wrapper
    """
    def __init__(self, api_key: str, model_name: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def chat(self, messages: list[dict[str, str]]) -> str:
        """
        Send a list of messages (system/user) to Gemini and return the generated text
        :param messages: List of {"role": ..., "content": ...} dicts
        :return: Response text from Gemini
        """
        # extract system instruction and user content
        system_instruction = next(
            (m['content'] for m in messages if m['role'] == 'system'),
            ""
        )
        # concatenate all user messages
        user_contents = [m['content'] for m in messages if m['role'] == 'user']
        user_prompt = "\n".join(user_contents)

        # call the generate_content endpoint
        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
            contents=user_prompt
        )
        return response.text

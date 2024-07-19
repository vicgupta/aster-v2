# pip install ollama
from ollama import Client


class OllamaModel:

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3"):
        self._host = host
        self._model = model
        self._ollama = Client(host)

    def ask(self, prompt: str, format: str = "", temperature: float = 0.5, context_window: int = 2048,):
        return self._ollama.chat(
            model=self._model,
            messages=prompt,
            format=format,
            options={
                "temperature": temperature,
                "num_ctx": context_window,
            },
        )["message"]["content"]


# pip install groq
# api keys at: https://console.groq.com/keys
# groq models = llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768
from groq import Groq


class GroqModel:

    def __init__(self, model: str = "llama3-8b-8192", api_key: str = "",):
        self._model = model
        self._api_key = api_key
        self._groq = Groq(api_key=self._api_key)

    def ask(self, prompt: str):
        return self._groq.chat.completions.create(
            messages=prompt, model=self._model,
        ).choices[0].message.content

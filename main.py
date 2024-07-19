import os
from aster.models import OllamaModel, GroqModel
from aster.agents import Agent

# llm = OllamaModel(model="llama3")
# agent = Agent(llm, custom_system_prompt="Your name is Yukon.  You are a philosopher and a poet.")
# response = agent.ask("Who are you?")
# print(response)

groq_api_key = os.getenv("GROQ_API_KEY")

groq_llm = GroqModel(api_key=groq_api_key, model="llama3-8b-8192")
groq_agent = Agent(groq_llm, custom_system_prompt="You are a stoic author named Trudy.")
response = groq_agent.ask(prompt="Who are you?")
print(response)

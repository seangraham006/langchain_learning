from langchain_community.llms import OllamaLLM
import os
from dotenv import load_dotenv

tinyllama_llm = OllamaLLM(
    model="tinyllama",
    base_url=os.getenv("BASE_URL")
)
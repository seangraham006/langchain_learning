from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv

load_dotenv()

TinyLlamaModel = OllamaLLM(
    model="tinyllama",
    base_url=os.getenv("BASE_URL")
)
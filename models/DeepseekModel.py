from langchain_community.llms import OllamaLLM
import os
from dotenv import load_dotenv

DeepseekModel = OllamaLLM(
    model="deepseek-r1:1.5b",
    base_url=os.getenv("BASE_URL")
)
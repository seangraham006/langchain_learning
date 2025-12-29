from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv

load_dotenv()

MistralModel = ChatMistralAI(
    model="mistral-medium",
    api_key=os.getenv("MISTRAL_API_KEY")
)
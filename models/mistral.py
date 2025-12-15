from langchain_community.chat_models import ChatMistralAI
import os
from dotenv import load_dotenv

mistral_llm = ChatMistralAI(
    model="mistral-medium",
    api_key=os.getenv("MISTRAL_API_KEY")
)
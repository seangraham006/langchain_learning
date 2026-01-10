from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv

load_dotenv()


def _create_mistral_model() -> ChatMistralAI:
    """
    Create and return a Mistral model instance.
    Validates that MISTRAL_API_KEY is set.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY environment variable is not set. "
            "Please set it in your .env file."
        )
    
    return ChatMistralAI(
        model="mistral-medium",
        api_key=api_key
    )


MistralModel = _create_mistral_model()
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_mistralai import ChatMistralAI

load_dotenv()

tinyllama_llm = OllamaLLM(
    model="tinyllama",
    base_url=os.getenv("BASE_URL")
)

deepseek_llm = OllamaLLM(
    model="deepseek-r1:1.5b",
    base_url=os.getenv("BASE_URL")
)

mistral_llm = ChatMistralAI(
    model="mistral-medium",
    api_key=os.getenv("MISTRAL_API_KEY")
)


print("Testing DeepSeek-R1 LLM")
response = deepseek_llm.invoke("Explain what LangChain does in one sentence.")
print(response)

print("\nTesting TinyLlama LLM")
response = tinyllama_llm.invoke("Explain what LangChain does in one sentence.")
print(response)

print("\nTesting Mistral LLM")
response = mistral_llm.invoke("Explain what LangChain does in one sentence.")
print(response)

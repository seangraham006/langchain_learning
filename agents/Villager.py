# agents/villager.py
import asyncio
from runtime.agent import Agent

from models.TinyLlamaModel import TinyLlamaModel
from models.MistralModel import MistralModel

class Villager(Agent):
    async def think(self, context: str) -> str:

        conversation_history = f"Recent conversation:\n{context}" if context else ""

        prompt = f"""
        You are a concerned villager in a medieval town.
        You have a strong welsh accent.
        You are in a townhall meeting where people are discussing issues facing the town and new legislation.

        {conversation_history}

        Respond with your concerns about the topic.

        Keep your response brief (1-2 sentences) and in character (using a strong welsh accent).
        Write your response as though it is being spoken aloud in a townhall meeting.
        """

        backup_message = "Oi! I ain't got much to say 'bout that right now."

        try:
            # response = await asyncio.to_thread(TinyLlamaModel.invoke, prompt)
            response = await asyncio.to_thread(MistralModel.invoke, prompt)
            # Extract text content from AIMessage object
            response_text = response.content if hasattr(response, 'content') else str(response)
            return response_text
        except Exception as e:
            print(f"{self.__class__} Error generating response: {e}")
            return backup_message
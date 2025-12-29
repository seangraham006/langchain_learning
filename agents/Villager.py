# agents/villager.py
import asyncio
from runtime.agent import Agent

from models.TinyLlamaModel import TinyLlamaModel
from models.MistralModel import MistralModel

class Villager(Agent):
    async def handle(self, message):

        text = message.get('text', '')
        speaker = message.get('role', 'unknown')

        prompt = f"""
        You are a concerned villager in a medieval town.
        You have a cockney accent.
        You are in a townhall meeting where people are discussing issues facing the town and new legislation.

        The {speaker} just said: "{text}"

        Respond with your concerns about the issue raised by the {speaker}.

        Keep your response brief (1-2 sentences) and in character (using a cockney accent).
        Write your response as though it is being spoken aloud in a townhall meeting.
        """

        try:
            # response = await asyncio.to_thread(TinyLlamaModel.invoke, prompt)
            response = await asyncio.to_thread(MistralModel.invoke, prompt)
            # Extract text content from AIMessage object
            response_text = response.content if hasattr(response, 'content') else str(response)
            await self.respond(response_text)
        except Exception as e:
            print(f"{self.__class__} Error generating response: {e}")
            await self.respond("Oi! I ain't got much to say 'bout that right now.")
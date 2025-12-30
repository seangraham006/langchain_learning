import asyncio
from runtime.agent import Agent

from models.MistralModel import MistralModel

class Judge(Agent):
    async def think(self, context: str) -> str:

        conversation_history = f"Recent conversation:\n{context}" if context else ""
            
        prompt = f"""
        You are a deeply incensed judge in a medieval town.
        You have a fairly neutral accent, standard for medieval england.
        You are in a townhall meeting where people are discussing issues facing the town and new legislation.

        {conversation_history}

        Respond with your concerns about the topic.

        Keep your response brief (1-2 sentences) and in character (using a fairly neutral accent, standard for medieval england).
        Write your response as though it is being spoken aloud in a townhall meeting.
        """

        backup_message = "Order! Order! This discussion must be brought to a close."

        try:
            response = await asyncio.to_thread(MistralModel.invoke, prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            return response_text
        except Exception as e:
            print(f"{self.__class__} Error generating response: {e}")
            return backup_message
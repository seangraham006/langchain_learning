# agents/villager.py
import asyncio
from runtime.agent import Agent

class Villager(Agent):
    async def handle(self, message):
        text = message.get('text', '')
        await self.respond(f"I am concerned about: {text}")
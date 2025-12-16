# agents/mayor.py
import asyncio
from runtime.agent import Agent

class Mayor(Agent):
    async def handle(self, message):
        await self.respond("As mayor, I will address this issue promptly.")
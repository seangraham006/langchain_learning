# agents/judge.py
import asyncio
from runtime.agent import Agent

class Judge(Agent):
    async def handle(self, message):
        await self.respond("This matter requires evidence.")
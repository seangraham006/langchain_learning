# agents/villager.py
import asyncio
from runtime.agent import Agent

class Villager(Agent):
    async def handle(self, message):
        print("Villager heard:", message)
        await asyncio.sleep(1)
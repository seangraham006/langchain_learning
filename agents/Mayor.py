# agents/mayor.py
import asyncio
from runtime.agent import Agent

class Mayor(Agent):
    async def handle(self, message):
        print("Mayor considers:", message)
        await asyncio.sleep(2)

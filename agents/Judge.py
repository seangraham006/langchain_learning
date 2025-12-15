# agents/judge.py
import asyncio
from runtime.agent import Agent

class Judge(Agent):
    async def handle(self, message):
        print("Judge evaluates:", message)
        await asyncio.sleep(3)
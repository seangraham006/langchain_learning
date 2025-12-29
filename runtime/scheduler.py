# runtime/scheduler.py
import asyncio
from agents.Villager import Villager
from agents.Mayor import Mayor
from agents.Judge import Judge

async def start_agents():
    agents = [
        Villager("villagers"),
        Mayor("mayor"),
        Judge("judge")
    ]

    await asyncio.gather(*(agent.run() for agent in agents))
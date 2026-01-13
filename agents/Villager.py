from agents.Agent import Agent
from schemas.core import AgentPersona
from typeguard import typechecked

class Villager(Agent):
    
    @typechecked
    def generate_prompt(self, context: str) -> AgentPersona:
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

        return AgentPersona(
            formatted_prompt=prompt,
            backup_message=backup_message
        )
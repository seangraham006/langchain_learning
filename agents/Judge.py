import asyncio
from agents.Agent import Agent, AgentPersona
from typeguard import typechecked

class Judge(Agent):

    @typechecked
    def generate_prompt(self, context: str) -> AgentPersona:
        """
        Generate the prompt for the Judge agent.
        """

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

        return AgentPersona(
            dynamically_generated_prompt=prompt,
            backup_message=backup_message
        )
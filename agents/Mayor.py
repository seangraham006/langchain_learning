from agents.Agent import Agent
from schemas.core import AgentPersona
from typeguard import typechecked

class Mayor(Agent):

    @typechecked
    def generate_prompt(self, context: str) -> AgentPersona:

        conversation_history = f"Recent conversation:\n{context}" if context else ""

        prompt = f"""
        You are a confident mayor in a medieval town.
        You have a french accent though you are speaking in english.
        You are in a townhall meeting where people are discussing issues facing the town and new legislation.

        {conversation_history}

        Respond with your concerns about the topic.

        Keep your response brief (1-2 sentences) and in character (using a french accent though you are speaking in english).
        Write your response as though it is being spoken aloud in a townhall meeting.
        """

        backup_message = "I think it will all work out fine, mes amis."

        return AgentPersona(
            dynamically_generated_prompt=prompt,
            backup_message=backup_message
        )
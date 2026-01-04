from pydantic import BaseModel

class StreamMessage(BaseModel):
    msg_id: str
    role: str
    text: str

class AgentPersona(BaseModel):
    dynamically_generated_prompt: str
    backup_message: str
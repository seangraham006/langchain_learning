from pydantic import BaseModel

class StreamMessage(BaseModel):
    msg_id: str
    role: str
    text: str

class AgentPersona(BaseModel):
    dynamically_generated_prompt: str
    backup_message: str

class SummaryRecord(BaseModel):
    stream_name: str
    start_msg_id: str
    end_msg_id: str
    summary_text: str
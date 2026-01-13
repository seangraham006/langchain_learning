from pydantic import BaseModel, Field

class StreamMessage(BaseModel):
    msg_id: str = Field(description="Redis stream message ID (timestamp-sequence format)")
    role: str = Field(description="Role/name of the agent or entity that created this message")
    text: str = Field(description="The content/body of the message")

class AgentPersona(BaseModel):
    formatted_prompt: str = Field(description="Formatted string that defines the agent's personality and behavior")
    backup_message: str = Field(description="Fallback message to use if the agent fails to generate a response")

class SummaryMetadata(BaseModel):
    """Summary metadata without embedding - used for lightweight queries."""
    stream_name: str = Field(description="Name of the Redis stream being summarized")
    start_msg_id: str = Field(description="First message ID included in this summary")
    end_msg_id: str = Field(description="Last message ID included in this summary")
    summary_text: str = Field(description="The AI generated summary text")

class SummaryRecord(BaseModel):
    """Complete summary record with required embedding - used for insertions."""
    stream_name: str = Field(description="Name of the Redis stream being summarized")
    start_msg_id: str = Field(description="First message ID included in this summary")
    end_msg_id: str = Field(description="Last message ID included in this summary")
    summary_text: str = Field(description="The AI generated summary text")
    embedding: bytes = Field(description="Serialized numpy array (768-dim) representing the semantic embedding of the summary")

from pydantic import BaseModel, Field

class StreamMessage(BaseModel):
    """Canonical schema for one Redis stream entry after parsing and validation."""
    msg_id: str = Field(description="Redis stream message ID (timestamp-sequence format)")
    role: str = Field(description="Role/name of the agent or entity that created this message")
    text: str = Field(description="The content/body of the message")

class AgentPersona(BaseModel):
    """Prompt configuration for an agent, including behavior instructions and fallback text."""
    formatted_prompt: str = Field(description="Formatted string that defines the agent's personality and behavior")
    backup_message: str = Field(description="Fallback message to use if the agent fails to generate a response")

class SummaryMetadata(BaseModel):
    """Lightweight summary record without vector bytes, used for listing and metadata queries."""
    stream_name: str = Field(description="Name of the Redis stream being summarized")
    start_msg_id: str = Field(description="First message ID included in this summary")
    end_msg_id: str = Field(description="Last message ID included in this summary")
    summary_text: str = Field(description="The AI generated summary text")

class SummaryRecord(BaseModel):
    """Full persisted summary payload, including embedding bytes required for storage and indexing."""
    stream_name: str = Field(description="Name of the Redis stream being summarized")
    start_msg_id: str = Field(description="First message ID included in this summary")
    end_msg_id: str = Field(description="Last message ID included in this summary")
    summary_text: str = Field(description="The AI generated summary text")
    embedding: bytes = Field(description="Serialized numpy array (768-dim) representing the semantic embedding of the summary")

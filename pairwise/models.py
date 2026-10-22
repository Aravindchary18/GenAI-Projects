from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Literal, Optional

Dataset = Literal["synthetic", "real"]

class ParticipantCreate(BaseModel):
    dataset: Dataset = "synthetic"
    display_name: str = Field(min_length=1, max_length=80)
    age: int = Field(ge=18, le=100)
    bio: str = Field(min_length=10, max_length=4000)
    interests: list[str] = Field(default_factory=list, max_length=30)
    hobbies: list[str] = Field(default_factory=list, max_length=30)
    values: list[str] = Field(default_factory=list, max_length=30)
    relationship_goals: list[str] = Field(default_factory=list, max_length=20)
    lifestyle: list[str] = Field(default_factory=list, max_length=30)
    preferences: dict = Field(default_factory=dict)
    linkedin_url: Optional[str] = None
    instagram_url: Optional[str] = None
    consent_confirmed: bool = False
    source_text: Optional[str] = Field(default=None, max_length=8000)

    @field_validator("linkedin_url")
    @classmethod
    def valid_linkedin(cls, v):
        if v and not (v.startswith("https://www.linkedin.com/") or v.startswith("https://linkedin.com/")):
            raise ValueError("Use a valid HTTPS LinkedIn profile URL.")
        return v

    @field_validator("instagram_url")
    @classmethod
    def valid_instagram(cls, v):
        if v and not (v.startswith("https://www.instagram.com/") or v.startswith("https://instagram.com/")):
            raise ValueError("Use a valid HTTPS Instagram profile URL.")
        return v

class AgentRunRequest(BaseModel):
    participant_a_id: str
    participant_b_id: str
    max_turns: int = Field(default=4, ge=2, le=10)

class AnalyzeRequest(BaseModel):
    participant_id: str

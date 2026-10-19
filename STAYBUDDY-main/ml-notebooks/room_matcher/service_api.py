from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="StayBuddy private roommate matcher", version="3.0")


class MatchRequest(BaseModel):
    subject: dict[str, Any]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    limit: int = 10


@app.get("/health")
def health():
    return {"status": "healthy", "service": "roommate-matcher", "sample_data": False}


@app.post("/match")
def match(request: MatchRequest):
    keys = ("university", "department", "study_preference", "food_preference")
    matches = []
    for candidate in request.candidates:
        signals = sum(1 for key in keys if request.subject.get(key) is not None and request.subject.get(key) == candidate.get(key))
        matches.append({"user_id": candidate["user_id"], "compatibility_signals": signals})
    matches.sort(key=lambda item: (-item["compatibility_signals"], item["user_id"]))
    return {"suggestions_only": True, "matches": matches[: max(1, min(request.limit, 20))]}

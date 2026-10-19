from typing import Any, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import llm_assistant

app = FastAPI(title="StayBuddy private recommendation and LLM RAG service", version="3.6")


class RecommendRequest(BaseModel):
    student_id: int
    student_profile: dict[str, Any] = Field(default_factory=dict)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    provider: Literal["ollama", "groq"] = "ollama"
    history: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    role: str = "guest"
    language: str = "en"
    intent: str = "general_support"
    retrieved_context: list[dict[str, Any]] = Field(default_factory=list)
    live_context: list[dict[str, Any]] = Field(default_factory=list, max_length=8)
    previous_plan: dict[str, Any] = Field(default_factory=dict)
    app_context: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "recommendation-chat",
        "generator": "ollama",
        "model": llm_assistant.MODEL,
        "data_source": "node_authorized_payload",
        "providers": {"ollama": True, "groq": bool(llm_assistant.os.getenv("GROQ_API_KEY", "").strip())},
    }


@app.post("/recommend")
def recommend(request: RecommendRequest):
    budget = request.constraints.get("budget_max")
    budget = float(budget) if budget not in (None, "") else None
    required = {str(value).lower() for value in request.constraints.get("required_amenities", [])}
    scored = []
    for candidate in request.candidates:
        amenities = {str(value).lower() for value in candidate.get("amenities") or []}
        prices = [float(value) for value in (candidate.get("single_room_price"), candidate.get("double_room_price"), candidate.get("dorm_room_price")) if value is not None]
        price = min(prices) if prices else None
        if budget is not None and (price is None or price > budget):
            continue
        if not required.issubset(amenities):
            continue
        rating = float(candidate.get("average_rating") or 0)
        availability = min(float(candidate.get("available_capacity") or 0), 20) / 20
        university_match = 1 if request.student_profile.get("university") and str(candidate.get("university", "")).lower() == str(request.student_profile["university"]).lower() else 0
        score = round(0.6 * rating / 5 + 0.25 * availability + 0.15 * university_match, 4)
        scored.append({"hostel_id": int(candidate["id"]), "hybrid_score": score, "cb_score": score, "cf_score": 0, "student_type": "database_profile", "alpha_used": 1.0})
    scored.sort(key=lambda item: (-item["hybrid_score"], item["hostel_id"]))
    return {"success": True, "recommendations": scored[: max(1, min(request.top_k, 20))], "student_type": "database_profile", "alpha_used": 1.0}


@app.get("/ready")
def ready():
    """Check model presence separately from liveness; does not run inference."""
    try:
        with llm_assistant.http.urlopen(llm_assistant.URL + "/api/tags", timeout=5) as response:
            models = llm_assistant.json.loads(response.read())["models"]
        if not any(m.get("name") == llm_assistant.MODEL for m in models):
            raise ValueError("model not installed")
        return {"status": "ready", "model": llm_assistant.MODEL}
    except Exception as exc:
        raise HTTPException(503, "Ollama or the configured model is unavailable") from exc


@app.post("/chat/plan")
def plan(request: ChatRequest):
    try:
        return llm_assistant.plan(request.model_dump())
    except Exception as exc:
        raise HTTPException(503, "Question interpretation temporarily unavailable") from exc


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        return llm_assistant.answer(request.model_dump())
    except Exception as exc:
        raise HTTPException(503, "The language model is temporarily unavailable") from exc

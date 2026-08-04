from __future__ import annotations


_DOMAIN_GROUP_KEYWORDS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "technology_data",
        frozenset(
            {
                "api", "app", "build", "camera", "code", "database",
                "data", "device", "digital", "email", "file", "interface",
                "mobile", "network", "networks", "pipeline", "printer",
                "sensor", "software", "storage", "sync", "technical", "web",
                "bug", "browser", "computing", "peripheral", "spreadsheet",
                "audio", "battery",
            }
        ),
    ),
    (
        "quantitative_reasoning",
        frozenset(
            {
                "arithmetic", "calculation", "calculus", "conversion",
                "equation", "logic", "logical", "math", "measurement",
                "probability", "proportion", "proportions", "rate",
                "sequence", "statistics", "statistical", "table", "threshold",
                "unit", "mathematics", "mechanical", "timeline",
            }
        ),
    ),
    (
        "education_research",
        frozenset(
            {
                "article", "course", "education", "experiment", "field",
                "historical", "lab", "learning", "lesson", "reading",
                "research", "science", "study", "survey", "training",
                "bibliographic", "grant", "tutorial", "grammar", "language",
            }
        ),
    ),
    (
        "governance_legal",
        frozenset(
            {
                "accessibility", "audit", "civic", "compliance",
                "consultation", "contract", "governance", "legal", "policy",
                "public", "regulation", "voting", "civics", "appeal",
            }
        ),
    ),
    (
        "finance_commerce",
        frozenset(
            {
                "account", "banking", "bill", "budget", "commerce", "cost",
                "discount", "finance", "invoice", "price", "procurement",
                "purchase", "retail", "returns", "shopping", "subscription",
                "subscriptions", "expense", "financial", "fraud", "receipt",
                "purchasing",
            }
        ),
    ),
    (
        "health_safety",
        frozenset(
            {
                "care", "emergency", "harm", "health", "healthcare",
                "medical", "privacy", "risk", "safe", "safety", "hazard",
                "harassment", "medication", "scam", "substance", "bite",
            }
        ),
    ),
    (
        "environment_energy",
        frozenset(
            {
                "climate", "ecology", "energy", "environment",
                "environmental", "sustainability", "utility", "ecological",
                "weather",
            }
        ),
    ),
    (
        "travel_logistics",
        frozenset(
            {
                "appointment", "appointments", "booking", "calendar", "delivery", "event",
                "logistics", "parcel", "relocation", "schedule", "transit",
                "transport", "travel", "vehicle", "venue", "shipment",
            }
        ),
    ),
    (
        "operations_projects",
        frozenset(
            {
                "customer", "design", "equipment", "handover", "incident",
                "maintenance", "meeting", "office", "operations", "product",
                "project", "release", "service", "specs", "support", "status",
                "work", "workplace", "inventory", "administrative", "document",
                "allocation", "application", "case", "ticket", "renewal",
            }
        ),
    ),
    (
        "communication_media",
        frozenset(
            {
                "argument", "caption", "claim", "communication", "copy",
                "draft", "explanation", "instructions", "interview", "media",
                "memo", "notice", "pronoun", "summary", "transcript",
                "writing", "brief", "feedback", "format", "reference", "form",
                "content", "instruction",
            }
        ),
    ),
    (
        "home_lifestyle",
        frozenset(
            {
                "community", "food", "home", "household", "meal", "personal",
                "recipe", "relationship", "repair", "coffee", "movie", "pizza",
                "restaurant", "housing",
            }
        ),
    ),
    (
        "interpersonal_wellbeing",
        frozenset(
            {
                "achievement", "afraid", "angry", "anticipating", "apprehensive",
                "ashamed", "caring", "confident", "creative", "devastated",
                "disappointed", "disgusted", "embarrassed", "excited", "faithful",
                "furious", "grateful", "grief", "guilty", "hopeful", "impressed",
                "jealous", "joyful", "lonely", "loss", "prepared", "proud",
                "rejection", "sad", "sentimental", "social", "stress", "terrified",
                "trusting",
            }
        ),
    ),
)


def domain_group(domain: str) -> str:
    """Map a detailed domain label to one stable, inspectable theme."""

    tokens = frozenset(
        token
        for token in domain.lower().replace("-", "_").split("_")
        if token
    )
    for group, keywords in _DOMAIN_GROUP_KEYWORDS:
        if tokens & keywords:
            return group
    return "general_cross_domain"

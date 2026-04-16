from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class ResearchState(TypedDict):
    # User input
    query: str
    requirements: str

    # Search phase
    search_conditions: Dict[str, Any]
    papers_metadata: List[Dict[str, Any]]
    search_approved: bool
    search_feedback: str

    # Reading phase
    parsed_papers: List[Dict[str, Any]]
    extracted_info: List[Dict[str, Any]]
    vectorstore_path: str
    rag_initialized: bool

    # Analysis phase
    topic_clusters: List[Dict[str, Any]]
    topic_analyses: List[Dict[str, Any]]
    global_insights: Dict[str, Any]

    # Writing phase
    outline: List[Dict[str, Any]]
    outline_approved: bool
    outline_feedback: str
    chapter_drafts: List[Dict[str, Any]]

    # Final output
    final_report: str


REQUIRED_STATE_KEYS = {
    "query",
    "requirements",
    "search_conditions",
    "papers_metadata",
    "search_approved",
    "search_feedback",
    "parsed_papers",
    "extracted_info",
    "vectorstore_path",
    "rag_initialized",
    "topic_clusters",
    "topic_analyses",
    "global_insights",
    "outline",
    "outline_approved",
    "outline_feedback",
    "chapter_drafts",
    "final_report",
}


def create_initial_state(query: str, requirements: str = "") -> ResearchState:
    """Create a complete ResearchState with safe defaults."""
    return ResearchState(
        query=query,
        requirements=requirements,
        search_conditions={},
        papers_metadata=[],
        search_approved=False,
        search_feedback="",
        parsed_papers=[],
        extracted_info=[],
        vectorstore_path="",
        rag_initialized=False,
        topic_clusters=[],
        topic_analyses=[],
        global_insights={},
        outline=[],
        outline_approved=False,
        outline_feedback="",
        chapter_drafts=[],
        final_report="",
    )


def validate_state_keys(state: Dict[str, Any]) -> List[str]:
    """Return a sorted list of missing required state keys."""
    return sorted(REQUIRED_STATE_KEYS.difference(state.keys()))

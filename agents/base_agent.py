"""
Base agent class — shared logic for all specialized agents.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from llm.base import BaseLLM, LLMResponse
from rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

# System instruction embedded in every agent prompt
ANTI_HALLUCINATION_INSTRUCTION = """
IMPORTANT RULES:
1. Only use information found in the provided context documents.
2. If the context does not contain the answer, say clearly: "I couldn't find reliable information about this in the available admission documents."
3. Do NOT invent fees, deadlines, eligibility criteria, scholarships, or any specific numbers.
4. Always cite which document/source supports your answer.
5. If partially answered, say what was found and what could not be confirmed.
""".strip()


@dataclass
class AgentResponse:
    """Structured response from any agent."""
    text: str
    agent_name: str
    sources: list[dict] = field(default_factory=list)
    category: str = "GENERAL"
    error: Optional[str] = None
    raw_llm_response: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.text)


class BaseAgent(ABC):
    """
    Abstract base for all specialized admission agents.
    Provides shared RAG retrieval and LLM generation logic.
    """

    name: str = "BaseAgent"

    def __init__(self, llm: BaseLLM, retriever: RAGRetriever):
        self.llm = llm
        self.retriever = retriever

    @abstractmethod
    def respond(
        self,
        query: str,
        student_profile: Optional[dict] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> AgentResponse:
        """
        Generate a response to the user's query.

        Args:
            query: The user's question.
            student_profile: Optional dict with student background info.
            conversation_history: Recent chat history for context.

        Returns:
            AgentResponse with text, sources, and metadata.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _retrieve_and_format(self, query: str, top_k: int = 4) -> tuple[str, list[dict]]:
        """
        Retrieve relevant chunks and format them into context + source list.

        Returns:
            (context_str, sources_list)
        """
        results = self.retriever.retrieve(query, top_k=top_k)
        context = self.retriever.format_context(results)
        sources = self.retriever.format_sources(results)
        return context, sources

    def _build_system_prompt(self, role_description: str) -> str:
        """Build a system prompt combining role + anti-hallucination rules."""
        return f"{role_description}\n\n{ANTI_HALLUCINATION_INSTRUCTION}"

    def _build_history_string(
        self, history: Optional[list[dict]], max_turns: int = 5
    ) -> str:
        """Convert conversation history to a prompt-friendly string."""
        if not history:
            return ""
        recent = history[-max_turns:]
        lines: list[str] = []
        for turn in recent:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            label = "Student" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    def _build_profile_string(self, profile: Optional[dict]) -> str:
        """Format student profile for inclusion in the prompt."""
        if not profile:
            return ""
        parts: list[str] = []
        field_labels = {
            "background": "Academic Background",
            "marks": "Marks/Percentage",
            "subjects": "Subjects Studied",
            "interests": "Interests",
            "preferred_course": "Preferred Course",
            "budget": "Budget",
            "scholarship_needed": "Scholarship Needed",
            "entrance_exam": "Entrance Exam Status",
        }
        for key, label in field_labels.items():
            val = profile.get(key, "")
            if val:
                parts.append(f"- {label}: {val}")
        if not parts:
            return ""
        return "Student Profile:\n" + "\n".join(parts)

    def _call_llm(self, prompt: str, system_prompt: str) -> LLMResponse:
        """Call the LLM with error handling."""
        try:
            return self.llm.generate(prompt, system_prompt=system_prompt)
        except Exception as e:
            logger.exception(f"LLM call failed in {self.name}: {e}")
            from llm.base import LLMResponse
            return LLMResponse(
                text="",
                model=getattr(self.llm, "model", "unknown"),
                provider="unknown",
                error=str(e),
            )

    def _no_info_response(self, topic: str) -> str:
        """Standard fallback when no relevant information is found."""
        return (
            f"I couldn't find reliable information about **{topic}** "
            "in the available admission documents.\n\n"
            "I recommend:\n"
            "- Visiting the official college website\n"
            "- Contacting the admissions office directly\n"
            "- Checking the official admission notification\n\n"
            "_Note: The knowledge base may need to be updated with the latest documents._"
        )

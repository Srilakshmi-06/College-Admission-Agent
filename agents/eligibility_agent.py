"""
Eligibility Agent — checks whether a student meets admission requirements
for specific courses based on retrieved eligibility criteria.
"""

import logging
from typing import Optional

from agents.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)

ROLE = """You are the Eligibility Agent for an AI College Admission System.
Your role is to help students understand whether they meet the admission requirements for specific courses.

When assessing eligibility:
1. Extract the eligibility criteria from the retrieved documents
2. Compare with the student's stated qualifications (if provided)
3. Clearly state: ELIGIBLE / LIKELY ELIGIBLE / MAY NOT BE ELIGIBLE / UNABLE TO DETERMINE
4. List any missing requirements the student should check
5. Mention entrance exams if referenced in the documents

CRITICAL: Never invent eligibility criteria, percentages, or requirements.
If the documents do not contain specific criteria, say so clearly and recommend the student contact the admissions office."""


class EligibilityAgent(BaseAgent):
    """
    Specialized agent for eligibility checking and requirement analysis.
    """

    name = "Eligibility Agent"

    def respond(
        self,
        query: str,
        student_profile: Optional[dict] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> AgentResponse:
        """Assess eligibility based on query, profile, and retrieved criteria."""

        # Build a targeted retrieval query
        retrieval_query = self._build_retrieval_query(query, student_profile)
        context, sources = self._retrieve_and_format(retrieval_query, top_k=4)

        profile_str = self._build_profile_string(student_profile)
        history_str = self._build_history_string(conversation_history)

        prompt_parts: list[str] = []

        if profile_str:
            prompt_parts.append(profile_str)
        if history_str:
            prompt_parts.append(f"Recent Conversation:\n{history_str}")

        prompt_parts.append(f"Eligibility Information from Knowledge Base:\n{context}")

        prompt_parts.append(
            f"Student Query: {query}\n\n"
            "Please provide an eligibility assessment with this structure:\n\n"
            "**Eligibility Status**: [ELIGIBLE / LIKELY ELIGIBLE / MAY NOT BE ELIGIBLE / "
            "UNABLE TO DETERMINE — NOT IN DOCUMENTS]\n\n"
            "**Requirements Found in Documents**: List all criteria mentioned\n\n"
            "**Assessment**: Compare student's qualifications with requirements "
            "(if profile information was provided)\n\n"
            "**What to Verify**: Items the student must confirm with the college\n\n"
            "**Note**: Only use information from the provided context. "
            "Do not invent any criteria."
        )

        prompt = "\n\n".join(prompt_parts)
        system_prompt = self._build_system_prompt(ROLE)

        llm_response = self._call_llm(prompt, system_prompt)

        if not llm_response.success:
            if "No relevant information" in context:
                return AgentResponse(
                    text=self._no_info_response("eligibility criteria for this course"),
                    agent_name=self.name,
                    sources=[],
                    category="ELIGIBILITY",
                )
            return AgentResponse(
                text="",
                agent_name=self.name,
                sources=sources,
                category="ELIGIBILITY",
                error=llm_response.error,
            )

        return AgentResponse(
            text=llm_response.text,
            agent_name=self.name,
            sources=sources,
            category="ELIGIBILITY",
        )

    def _build_retrieval_query(self, query: str, profile: Optional[dict]) -> str:
        """Build a retrieval query for eligibility information."""
        parts: list[str] = [query, "eligibility requirements admission criteria"]
        if profile:
            if profile.get("marks"):
                parts.append(f"minimum marks percentage {profile['marks']}")
            if profile.get("subjects"):
                parts.append(f"required subjects {profile['subjects']}")
        return " ".join(parts)

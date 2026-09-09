"""
Course Recommendation Agent — recommends suitable courses based on
the student's academic background, interests, and retrieved course info.
"""

import logging
from typing import Optional

from agents.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)

ROLE = """You are the Course Recommendation Agent for an AI College Admission System.
Your role is to recommend suitable courses to students based on:
1. Their academic background and marks
2. Their stated interests
3. The course information retrieved from the knowledge base

Always:
- Explain WHY a course is recommended for this student
- List what the student should verify before applying (eligibility, deadlines, documents)
- Use structured formatting with clear headings
- Only mention courses found in the retrieved context
- Never invent courses, entry requirements, or seat availability"""


class CourseAgent(BaseAgent):
    """
    Specialized agent for course discovery and personalized recommendations.
    """

    name = "Course Recommendation Agent"

    def respond(
        self,
        query: str,
        student_profile: Optional[dict] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> AgentResponse:
        """Recommend courses based on the student's profile and query."""

        # Enrich the retrieval query with profile context
        retrieval_query = self._build_retrieval_query(query, student_profile)
        context, sources = self._retrieve_and_format(retrieval_query, top_k=5)

        profile_str = self._build_profile_string(student_profile)
        history_str = self._build_history_string(conversation_history)

        prompt_parts: list[str] = []

        if profile_str:
            prompt_parts.append(profile_str)
        if history_str:
            prompt_parts.append(f"Recent Conversation:\n{history_str}")

        prompt_parts.append(f"Available Course Information from Knowledge Base:\n{context}")

        prompt_parts.append(
            f"Student Request: {query}\n\n"
            "Based on the student's profile and the retrieved course information, "
            "provide a structured recommendation:\n\n"
            "1. **Recommended Course(s)** — List with brief description\n"
            "2. **Why These Courses Suit You** — Match with student background/interests\n"
            "3. **Before You Apply** — What to verify (eligibility, deadline, documents)\n"
            "4. **Additional Options** — Other courses worth considering, if available\n\n"
            "Only recommend courses mentioned in the retrieved information. "
            "If insufficient information is available, say so clearly."
        )

        prompt = "\n\n".join(prompt_parts)
        system_prompt = self._build_system_prompt(ROLE)

        llm_response = self._call_llm(prompt, system_prompt)

        if not llm_response.success:
            if "No relevant information" in context:
                return AgentResponse(
                    text=self._no_info_response("course information"),
                    agent_name=self.name,
                    sources=[],
                    category="COURSE",
                )
            return AgentResponse(
                text="",
                agent_name=self.name,
                sources=sources,
                category="COURSE",
                error=llm_response.error,
            )

        return AgentResponse(
            text=llm_response.text,
            agent_name=self.name,
            sources=sources,
            category="COURSE",
        )

    def _build_retrieval_query(self, query: str, profile: Optional[dict]) -> str:
        """Augment the query with profile signals for better retrieval."""
        extra: list[str] = [query]
        if profile:
            if profile.get("interests"):
                extra.append(f"courses related to {profile['interests']}")
            if profile.get("subjects"):
                extra.append(f"courses for students with {profile['subjects']} background")
            if profile.get("preferred_course"):
                extra.append(profile["preferred_course"])
        return " ".join(extra)

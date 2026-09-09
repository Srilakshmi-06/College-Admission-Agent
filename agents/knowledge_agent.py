"""
Admission Knowledge Agent — general admission Q&A with RAG retrieval.
Handles: FAQ, deadlines, general inquiries, and anything not covered
by a more specialized agent.
"""

import logging
from typing import Optional

from agents.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)

ROLE = """You are the Admission Knowledge Agent for an AI College Admission System.
Your role is to answer general admission questions accurately using only the provided context documents.
You handle questions about admission policies, important dates, deadlines, and general FAQs.
Be helpful, clear, and concise. Structure your answer with headers when it improves readability."""


class KnowledgeAgent(BaseAgent):
    """
    General-purpose admission knowledge agent.
    Uses RAG to answer a wide range of admission questions.
    """

    name = "Admission Knowledge Agent"

    def respond(
        self,
        query: str,
        student_profile: Optional[dict] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> AgentResponse:
        """Answer a general admission question using the knowledge base."""

        context, sources = self._retrieve_and_format(query, top_k=4)

        profile_str = self._build_profile_string(student_profile)
        history_str = self._build_history_string(conversation_history)

        prompt_parts: list[str] = []

        if profile_str:
            prompt_parts.append(profile_str)

        if history_str:
            prompt_parts.append(f"Recent Conversation:\n{history_str}")

        prompt_parts.append(
            f"Retrieved Admission Information:\n{context}"
        )
        prompt_parts.append(
            f"Student Question: {query}\n\n"
            "Provide a clear, accurate answer based only on the information above. "
            "Include source references where possible."
        )

        prompt = "\n\n".join(prompt_parts)
        system_prompt = self._build_system_prompt(ROLE)

        llm_response = self._call_llm(prompt, system_prompt)

        if not llm_response.success:
            # Check if no context was found
            if "No relevant information" in context:
                return AgentResponse(
                    text=self._no_info_response(query),
                    agent_name=self.name,
                    sources=[],
                    category="GENERAL",
                )
            return AgentResponse(
                text="",
                agent_name=self.name,
                sources=sources,
                category="GENERAL",
                error=llm_response.error,
            )

        return AgentResponse(
            text=llm_response.text,
            agent_name=self.name,
            sources=sources,
            category="GENERAL",
            raw_llm_response=llm_response.text,
        )

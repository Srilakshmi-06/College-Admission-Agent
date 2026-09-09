"""
Application Guidance Agent — guides students through the application process,
required documents, deadlines, and step-by-step procedures.
"""

import logging
from typing import Optional

from agents.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)

ROLE = """You are the Application Guidance Agent for an AI College Admission System.
Your role is to guide students through the college application process.

You help with:
- Step-by-step application procedures
- Required documents and checklists
- Important deadlines and dates
- How to submit applications (online/offline)
- What to expect after applying

Always structure your answers clearly with numbered steps and checklists.
Only provide information found in the retrieved documents.
If specific procedures are not in the documents, tell the student to contact the admissions office."""


class ApplicationAgent(BaseAgent):
    """
    Specialized agent for application process guidance and document checklists.
    """

    name = "Application Guidance Agent"

    def respond(
        self,
        query: str,
        student_profile: Optional[dict] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> AgentResponse:
        """Provide step-by-step application guidance."""

        retrieval_query = f"{query} application process documents required steps"
        context, sources = self._retrieve_and_format(retrieval_query, top_k=5)

        profile_str = self._build_profile_string(student_profile)
        history_str = self._build_history_string(conversation_history)

        prompt_parts: list[str] = []

        if profile_str:
            prompt_parts.append(profile_str)
        if history_str:
            prompt_parts.append(f"Recent Conversation:\n{history_str}")

        prompt_parts.append(f"Application Information from Knowledge Base:\n{context}")

        prompt_parts.append(
            f"Student Query: {query}\n\n"
            "Provide a clear, actionable application guide with:\n\n"
            "**Application Steps** (numbered list)\n\n"
            "**Required Documents Checklist** (if relevant)\n\n"
            "**Important Deadlines** (only if found in the documents)\n\n"
            "**Tips & Reminders**\n\n"
            "Use only information from the provided documents. "
            "Mark any item as '⚠️ Please verify with college' if it was not found in the context."
        )

        prompt = "\n\n".join(prompt_parts)
        system_prompt = self._build_system_prompt(ROLE)

        llm_response = self._call_llm(prompt, system_prompt)

        if not llm_response.success:
            if "No relevant information" in context:
                return AgentResponse(
                    text=self._no_info_response("application procedure and document requirements"),
                    agent_name=self.name,
                    sources=[],
                    category="APPLICATION",
                )
            return AgentResponse(
                text="",
                agent_name=self.name,
                sources=sources,
                category="APPLICATION",
                error=llm_response.error,
            )

        return AgentResponse(
            text=llm_response.text,
            agent_name=self.name,
            sources=sources,
            category="APPLICATION",
        )

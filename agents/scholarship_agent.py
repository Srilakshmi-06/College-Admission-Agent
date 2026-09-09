"""
Scholarship & Fee Agent — retrieves fee structures and scholarship information.
Handles both tuition fee queries and financial aid / scholarship questions.
"""

import logging
from typing import Optional

from agents.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)

ROLE = """You are the Scholarship and Fee Agent for an AI College Admission System.
Your role is to provide accurate information about:
- Tuition and other fees
- Scholarship opportunities
- Financial assistance programs
- Fee payment schedules
- Scholarship eligibility and application procedures

CRITICAL RULES:
- Never invent fee amounts, scholarship values, or eligibility percentages
- Only quote figures that appear verbatim in the retrieved documents
- If fee/scholarship details are not in the documents, state this clearly
- Suggest the student contact the finance office for up-to-date fee information"""


class ScholarshipAgent(BaseAgent):
    """
    Specialized agent for fees and scholarship information retrieval.
    """

    name = "Scholarship & Fee Agent"

    def respond(
        self,
        query: str,
        student_profile: Optional[dict] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> AgentResponse:
        """Retrieve and present fee and scholarship information."""

        retrieval_query = f"{query} fees scholarship financial aid tuition"
        context, sources = self._retrieve_and_format(retrieval_query, top_k=5)

        profile_str = self._build_profile_string(student_profile)
        history_str = self._build_history_string(conversation_history)

        prompt_parts: list[str] = []

        if profile_str:
            prompt_parts.append(profile_str)
        if history_str:
            prompt_parts.append(f"Recent Conversation:\n{history_str}")

        prompt_parts.append(f"Fee and Scholarship Information from Knowledge Base:\n{context}")

        prompt_parts.append(
            f"Student Query: {query}\n\n"
            "Provide a structured answer covering:\n\n"
            "**Fee Structure** (only figures found in documents)\n\n"
            "**Available Scholarships** (name, eligibility, amount — only from documents)\n\n"
            "**How to Apply for Scholarship** (if available in documents)\n\n"
            "**Financial Assistance Options** (if mentioned)\n\n"
            "**Important** ⚠️: If specific amounts are not in the documents, "
            "state 'Not available in current documents — please verify with the college finance office.' "
            "Never guess or invent any financial figures."
        )

        prompt = "\n\n".join(prompt_parts)
        system_prompt = self._build_system_prompt(ROLE)

        llm_response = self._call_llm(prompt, system_prompt)

        if not llm_response.success:
            if "No relevant information" in context:
                return AgentResponse(
                    text=self._no_info_response("fee structure and scholarship details"),
                    agent_name=self.name,
                    sources=[],
                    category="SCHOLARSHIP",
                )
            return AgentResponse(
                text="",
                agent_name=self.name,
                sources=sources,
                category="SCHOLARSHIP",
                error=llm_response.error,
            )

        return AgentResponse(
            text=llm_response.text,
            agent_name=self.name,
            sources=sources,
            category="SCHOLARSHIP",
        )

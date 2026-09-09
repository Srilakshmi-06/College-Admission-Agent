"""
Agent Coordinator — orchestrates multiple specialized agents for
complex queries that require information from several domains.
"""

import logging
from typing import Optional

from agents.base_agent import AgentResponse
from agents.router import QueryRouter, QueryCategory
from agents.knowledge_agent import KnowledgeAgent
from agents.course_agent import CourseAgent
from agents.eligibility_agent import EligibilityAgent
from agents.application_agent import ApplicationAgent
from agents.scholarship_agent import ScholarshipAgent
from llm.base import BaseLLM
from rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Coordinates query routing and multi-agent collaboration.
    Routes single-domain queries to the appropriate specialist.
    Merges multi-domain responses into a coherent answer.
    """

    def __init__(self, llm: BaseLLM, retriever: RAGRetriever):
        self.llm = llm
        self.retriever = retriever
        self.router = QueryRouter()

        # Initialize all specialized agents
        self._agents = {
            "knowledge": KnowledgeAgent(llm, retriever),
            "course": CourseAgent(llm, retriever),
            "eligibility": EligibilityAgent(llm, retriever),
            "application": ApplicationAgent(llm, retriever),
            "scholarship": ScholarshipAgent(llm, retriever),
        }

    def process(
        self,
        query: str,
        student_profile: Optional[dict] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> tuple[AgentResponse, QueryCategory]:
        """
        Process a user query end-to-end:
        classify → route → retrieve → generate response.

        Args:
            query: The student's question.
            student_profile: Optional profile dict.
            conversation_history: Recent chat history.

        Returns:
            Tuple of (AgentResponse, QueryCategory)
        """
        category = self.router.classify(query)
        logger.info(f"Query classified as: {category.value}")

        agent_names = self.router.get_agent_names(category)

        if len(agent_names) == 1:
            # Single agent
            agent = self._agents.get(agent_names[0], self._agents["knowledge"])
            response = agent.respond(query, student_profile, conversation_history)
            return response, category

        # Multi-agent collaboration
        return self._multi_agent_respond(
            query, agent_names, student_profile, conversation_history
        ), category

    def _multi_agent_respond(
        self,
        query: str,
        agent_names: list[str],
        student_profile: Optional[dict],
        conversation_history: Optional[list[dict]],
    ) -> AgentResponse:
        """
        Gather responses from multiple agents and merge them.
        """
        logger.info(f"Multi-agent collaboration: {agent_names}")

        sub_responses: list[AgentResponse] = []
        all_sources: list[dict] = []
        seen_source_keys: set[str] = set()

        for name in agent_names:
            agent = self._agents.get(name)
            if not agent:
                continue
            resp = agent.respond(query, student_profile, conversation_history)
            if resp.success:
                sub_responses.append(resp)
                for src in resp.sources:
                    key = f"{src['source']}_p{src.get('page','')}"
                    if key not in seen_source_keys:
                        seen_source_keys.add(key)
                        all_sources.append(src)

        if not sub_responses:
            # All agents failed — return a generic fallback
            fallback = self._agents["knowledge"]
            resp = fallback.respond(query, student_profile, conversation_history)
            return resp

        if len(sub_responses) == 1:
            r = sub_responses[0]
            r.sources = all_sources
            return r

        # Merge multiple responses into a unified answer
        merged_text = self._merge_responses(query, sub_responses, student_profile)
        return AgentResponse(
            text=merged_text,
            agent_name="Multi-Agent Collaboration",
            sources=all_sources,
            category="MULTI",
        )

    def _merge_responses(
        self,
        query: str,
        responses: list[AgentResponse],
        student_profile: Optional[dict],
    ) -> str:
        """
        Use the LLM to synthesize multiple agent responses into one coherent answer.
        """
        sections: list[str] = []
        for resp in responses:
            sections.append(f"[{resp.agent_name}]\n{resp.text}")

        combined = "\n\n---\n\n".join(sections)

        profile_info = ""
        if student_profile:
            items = [f"{k}: {v}" for k, v in student_profile.items() if v]
            if items:
                profile_info = "Student Profile: " + ", ".join(items) + "\n\n"

        synthesis_prompt = (
            f"{profile_info}"
            f"Student Question: {query}\n\n"
            f"Information gathered from multiple specialized agents:\n\n{combined}\n\n"
            "Create a single, coherent, well-structured response that:\n"
            "1. Directly addresses the student's question\n"
            "2. Integrates the information from all agents\n"
            "3. Avoids repetition\n"
            "4. Uses clear headings for each topic area\n"
            "5. Only includes information supported by the agents above"
        )

        system_prompt = (
            "You are an AI College Admission Coordinator. "
            "Synthesize the provided agent responses into a single, helpful answer. "
            "Do not add any new information not present in the provided agent responses."
        )

        llm_response = self._call_llm(synthesis_prompt, system_prompt)
        if llm_response.success:
            return llm_response.text

        # Fallback: concatenate with headers
        parts: list[str] = []
        for resp in responses:
            parts.append(f"### {resp.agent_name}\n\n{resp.text}")
        return "\n\n---\n\n".join(parts)

    def _call_llm(self, prompt: str, system_prompt: str):
        """Delegate LLM call through the first available agent."""
        try:
            return self.llm.generate(prompt, system_prompt=system_prompt)
        except Exception as e:
            logger.exception(f"Coordinator LLM call failed: {e}")
            from llm.base import LLMResponse
            return LLMResponse(text="", model="unknown", provider="unknown", error=str(e))

    def get_agent_status(self) -> dict[str, str]:
        """Return the names and descriptions of all available agents."""
        return {
            "knowledge": "Admission Knowledge Agent — General Q&A",
            "course": "Course Recommendation Agent — Personalized course advice",
            "eligibility": "Eligibility Agent — Eligibility checking",
            "application": "Application Guidance Agent — How to apply",
            "scholarship": "Scholarship & Fee Agent — Fees and financial aid",
        }

"""Agents package."""
from agents.router import QueryRouter, QueryCategory
from agents.base_agent import BaseAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.course_agent import CourseAgent
from agents.eligibility_agent import EligibilityAgent
from agents.application_agent import ApplicationAgent
from agents.scholarship_agent import ScholarshipAgent
from agents.coordinator import AgentCoordinator

__all__ = [
    "QueryRouter",
    "QueryCategory",
    "BaseAgent",
    "KnowledgeAgent",
    "CourseAgent",
    "EligibilityAgent",
    "ApplicationAgent",
    "ScholarshipAgent",
    "AgentCoordinator",
]

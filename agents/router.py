"""
Query router — classifies user queries into categories and routes
them to the appropriate specialized agent.
"""

import re
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QueryCategory(str, Enum):
    COURSE = "COURSE"
    ELIGIBILITY = "ELIGIBILITY"
    FEES = "FEES"
    APPLICATION = "APPLICATION"
    DOCUMENTS = "DOCUMENTS"
    DEADLINE = "DEADLINE"
    SCHOLARSHIP = "SCHOLARSHIP"
    FAQ = "FAQ"
    GENERAL = "GENERAL"
    MULTI = "MULTI"          # requires collaboration across agents


# Keyword patterns for each category (ordered by priority)
_ROUTING_RULES: list[tuple[QueryCategory, list[str]]] = [
    (QueryCategory.ELIGIBILITY, [
        r"\beligib", r"\bqualif", r"\brequirement", r"\bminimum mark",
        r"\bpercentage required", r"\bcan i (apply|join|get admitted)", r"\bam i eligible",
        r"\bentrance exam", r"\bcut.?off", r"\bscored?\b",
    ]),
    (QueryCategory.SCHOLARSHIP, [
        r"\bscholarship", r"\bfinancial aid", r"\bgrant", r"\bstipend",
        r"\bfee waiver", r"\bfree (education|course)", r"\bmerit",
    ]),
    (QueryCategory.FEES, [
        r"\bfee", r"\btuition", r"\bcost", r"\bprice", r"\bpay",
        r"\bexpens", r"\baffordabl", r"\bbudget",
    ]),
    (QueryCategory.DEADLINE, [
        r"\bdeadline", r"\blast date", r"\bdue date", r"\bwhen (is|are|do)",
        r"\bcalendar", r"\bschedule", r"\bimportant date", r"\bregistration date",
    ]),
    (QueryCategory.DOCUMENTS, [
        r"\bdocument", r"\bcertificate", r"\bmark.?sheet", r"\btranscript",
        r"\bid proof", r"\bphoto", r"\bcharacter certificate",
        r"\bwhat (do|should) i (submit|bring|need|upload)",
    ]),
    (QueryCategory.APPLICATION, [
        r"\bhow (do|to|can) (i |one |we )?apply", r"\bapplication (process|procedure|form|step)",
        r"\bapply (for|to)\b", r"\bsubmit (application|form)",
        r"\bonline (application|registration)", r"\bapplication portal",
        r"\bhow to apply\b", r"\bapplication procedure\b",
    ]),
    (QueryCategory.COURSE, [
        r"\bcourse", r"\bprogram", r"\bdegree", r"\bb\.?tech", r"\bb\.?sc",
        r"\bm\.?tech", r"\bmba", r"\bm\.?sc", r"\bwhich (course|program|stream)",
        r"\bavailable (courses|programs)", r"\bspecialization",
    ]),
]

_MULTI_SIGNALS: list[str] = [
    r"\band\b.*\band\b",          # "fees and eligibility and courses"
    r"\balso\b",
    r"\bas well as\b",
]


class QueryRouter:
    """
    Routes user queries to the most appropriate specialized agent(s).
    """

    def classify(self, query: str) -> QueryCategory:
        """
        Classify a query into a single primary category.

        Args:
            query: Raw user input text.

        Returns:
            QueryCategory enum value.
        """
        q = query.lower()

        # Check for multi-category query first
        matched_categories: list[QueryCategory] = []
        for category, patterns in _ROUTING_RULES:
            if any(re.search(p, q) for p in patterns):
                matched_categories.append(category)

        if len(matched_categories) == 0:
            return QueryCategory.GENERAL

        if len(matched_categories) == 1:
            return matched_categories[0]

        # With 2+ matches, use priority ordering from _ROUTING_RULES.
        # High-specificity categories override lower-specificity ones.
        # ELIGIBILITY and SCHOLARSHIP are more specific than COURSE.
        PRIORITY_ORDER = [
            QueryCategory.ELIGIBILITY,
            QueryCategory.SCHOLARSHIP,
            QueryCategory.FEES,
            QueryCategory.DEADLINE,
            QueryCategory.DOCUMENTS,
            QueryCategory.APPLICATION,
            QueryCategory.COURSE,
            QueryCategory.FAQ,
            QueryCategory.GENERAL,
        ]

        # Only treat as MULTI when genuinely different top-level intents are present
        # (e.g., someone asking about BOTH fees AND eligibility AND courses)
        high_level_matches = set(matched_categories)
        # Remove COURSE if query is primarily about a different category
        # (e.g., "eligible for B.Tech" should stay ELIGIBILITY)
        if len(high_level_matches) == 2:
            cats = high_level_matches
            # If the only pairing is COURSE + something else, don't escalate to MULTI.
            # MULTI should require at least 3 distinct major categories
            # OR two clearly orthogonal ones (e.g., ELIGIBILITY + FEES)
            orthogonal_pairs = {
                frozenset({QueryCategory.ELIGIBILITY, QueryCategory.FEES}),
                frozenset({QueryCategory.ELIGIBILITY, QueryCategory.SCHOLARSHIP}),
                frozenset({QueryCategory.COURSE, QueryCategory.FEES}),
                frozenset({QueryCategory.COURSE, QueryCategory.SCHOLARSHIP}),
            }
            if frozenset(cats) in orthogonal_pairs:
                return QueryCategory.MULTI
            # Otherwise return highest priority match
            for cat in PRIORITY_ORDER:
                if cat in cats:
                    return cat

        if len(high_level_matches) >= 3:
            return QueryCategory.MULTI

        # Fall through: return the first match in priority order
        for cat in PRIORITY_ORDER:
            if cat in matched_categories:
                return cat

        return matched_categories[0]

    def get_agent_names(self, category: QueryCategory) -> list[str]:
        """Return the list of agent names required for a category."""
        mapping: dict[QueryCategory, list[str]] = {
            QueryCategory.COURSE: ["course"],
            QueryCategory.ELIGIBILITY: ["eligibility"],
            QueryCategory.FEES: ["scholarship"],      # ScholarshipAgent handles fees too
            QueryCategory.APPLICATION: ["application"],
            QueryCategory.DOCUMENTS: ["application"],
            QueryCategory.DEADLINE: ["knowledge"],
            QueryCategory.SCHOLARSHIP: ["scholarship"],
            QueryCategory.FAQ: ["knowledge"],
            QueryCategory.GENERAL: ["knowledge"],
            QueryCategory.MULTI: ["course", "eligibility", "scholarship"],
        }
        return mapping.get(category, ["knowledge"])

    def route_description(self, category: QueryCategory) -> str:
        """Human-readable routing description for UI display."""
        desc = {
            QueryCategory.COURSE: "🎓 Course Recommendation Agent",
            QueryCategory.ELIGIBILITY: "✅ Eligibility Agent",
            QueryCategory.FEES: "💰 Scholarship & Fee Agent",
            QueryCategory.APPLICATION: "📋 Application Guidance Agent",
            QueryCategory.DOCUMENTS: "📄 Application Guidance Agent",
            QueryCategory.DEADLINE: "📅 Admission Knowledge Agent",
            QueryCategory.SCHOLARSHIP: "💰 Scholarship & Fee Agent",
            QueryCategory.FAQ: "🤖 Admission Knowledge Agent",
            QueryCategory.GENERAL: "🤖 Admission Knowledge Agent",
            QueryCategory.MULTI: "🤝 Multi-Agent Collaboration",
        }
        return desc.get(category, "🤖 Admission Knowledge Agent")

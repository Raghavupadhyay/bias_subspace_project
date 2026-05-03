"""
Custom prompts (less constrained than WinoBias).
Useful for generalization tests.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class CustomPrompt:
    text: str
    profession: str


PROFESSIONS = [
    "doctor", "engineer", "teacher", "nurse",
    "scientist", "manager", "chef"
]


TEMPLATES = [
    "The {profession} said that",
    "Yesterday the {profession} mentioned",
    "I met a {profession} and",
    "The {profession} believes that",
]


def generate_custom_prompts() -> List[CustomPrompt]:
    prompts = []
    for p in PROFESSIONS:
        for t in TEMPLATES:
            prompts.append(CustomPrompt(
                text=t.format(profession=p),
                profession=p
            ))
    return prompts
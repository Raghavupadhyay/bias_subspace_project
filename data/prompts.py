# experiemnt 001 simple dataset
# TEMPLATES = [
#     "The doctor said that ___ was tired.",
#     "The nurse said that ___ was tired.",
#     "The engineer said that ___ was late.",
#     "The teacher said that ___ was helpful.",
#     "The manager said that ___ was confident.",
#     "The scientist said that ___ was brilliant.",
# ]

# def build_prompts():
#     return [t.replace("___", "") for t in TEMPLATES]


# experiment 002 more datset 

import random

PROFESSIONS = [
    "doctor", "nurse", "engineer", "teacher", "scientist",
    "manager", "lawyer", "mechanic", "designer", "chef",
    "pilot", "assistant", "clerk", "developer", "analyst"
]

TEMPLATES = [
    "The {profession} said that ___ was tired.",
    "The {profession} mentioned that ___ was late.",
    "The {profession} believes ___ is skilled.",
    "The {profession} thinks ___ did a great job.",
    "The {profession} reported that ___ was absent.",
    "The {profession} noticed that ___ was confident."
]

def build_prompts(n=100, seed=42):
    random.seed(seed)
    prompts = []

    for _ in range(n):
        prof = random.choice(PROFESSIONS)
        temp = random.choice(TEMPLATES)
        prompts.append(temp.format(profession=prof).replace("___", ""))

    return prompts
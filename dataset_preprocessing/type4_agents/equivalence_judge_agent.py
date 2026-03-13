from pydantic import BaseModel, Field
from pydantic_ai import Agent


SYSTEM_PROMPT = """Act as an expert software engineer and program analysis specialist with deep expertise in semantic equivalence,
refactoring correctness, and vulnerability-preserving code transformations.
"""


class EquivalenceJudgementOutput(BaseModel):
    semantically_equivalent: bool = Field(
        description="True if original and transformed functions are semantically equivalent."
    )
    justification: str = Field(
        description="Short technical justification for the decision."
    )


equivalence_judge_agent = Agent(
    "gateway/openai:gpt-5.1",
    output_type=EquivalenceJudgementOutput,
    system_prompt=SYSTEM_PROMPT,
)

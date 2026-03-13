from pydantic import BaseModel, Field
from pydantic_ai import Agent


SYSTEM_PROMPT = """Act as an expert software engineer specializing in program transformation and semantic-preserving code rewriting.
"""


class Type4TransformationOutput(BaseModel):
    transformed_function: str = Field(
        description="Function source code after applying a Type-4 transformation."
    )
    transformation_description: str = Field(
        description="Short description of the applied transformation."
    )


type4_transform_agent = Agent(
    "gateway/openai:gpt-5.1",
    output_type=Type4TransformationOutput,
    system_prompt=SYSTEM_PROMPT,
)

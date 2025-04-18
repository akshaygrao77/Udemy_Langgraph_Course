from typing import List

from langchain_core.pydantic_v1 import BaseModel,Field

class Reflection(BaseModel):
    missing:str = Field(description="Critique of what is missing.")
    # Unnecessary information
    superfluous:str = Field(description="Critique of what is superfluous.")


class AnswerQuestion(BaseModel):
    """ Answer the question"""
    answer : str = Field(description="~250 word detailed answer to the question.")
    reflection:Reflection = Field(description="Your reflection for the initial question.")
    search_queries:List[str] = Field(description="1-3 search queries for research improvements to address the critique of your current answer.")

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question"""

    references:List[str] = Field(description="Citations motivating your updated answer.")

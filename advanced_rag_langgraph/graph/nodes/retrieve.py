from typing import Any,Dict

from graph.state import GraphState
from ingestion import retriever

def retrieve(state:GraphState)->Dict[str,Any]:
    print("----RETRIEVING----")
    question:str = state["question"]

    documents = retriever.invoke(question)
    # Question was rewritten just for safety
    return {"documents":documents,"question":question}
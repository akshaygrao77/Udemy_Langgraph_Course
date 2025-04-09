from typing import Any,Dict

from graph.state import GraphState
from graph.chains.generation import generation_chain

def generate_answers(state:GraphState)->Dict[str,Any]:
    print("--- GENERATE ANSWERS-----")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context":documents,"question":question})

    return {"documents":documents,"question":question,"generation":generation}
from dotenv import load_dotenv

load_dotenv()


import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    aggregate: Annotated[list, operator.add]


class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        import time

        time.sleep(1)
        print(f"Adding {self._value} to {state['aggregate']}")
        return {"aggregate": [self._value]}


builder = StateGraph(State)
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_edge(START, "a")
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("b2", ReturnNodeValue("I'm B2"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "b2")
builder.add_edge(["b2", "c"], "d")
builder.add_edge("d", END)
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="async.png")

if __name__ == "__main__":
    print("Helo Async Graph")
    graph.invoke({"aggregate": []}, {"configurable": {"thread_id": "foo"}})

# Adding I'm A to []
# Adding I'm C to ["I'm A"]
# Adding I'm B to ["I'm A"]
# Adding I'm B2 to ["I'm A", "I'm B", "I'm C"]
# Adding I'm D to ["I'm A", "I'm B", "I'm C", "I'm B2"]

# Above, node B,C ran simultaneously. Proof is that the state list has only A for both of them.
# Above, node D executed only after two parallel branches without needed to have wait commands. Proof is the list in D has all dependencies [B,B2,C]
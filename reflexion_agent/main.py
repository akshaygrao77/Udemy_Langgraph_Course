from dotenv import load_dotenv
load_dotenv()

from typing import List

from langchain_core.messages import BaseMessage, ToolMessage,HumanMessage,AIMessage
from langgraph.graph import END, MessageGraph

from chains import revisor,first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2

builder = MessageGraph()

builder.add_node("draft",first_responder)
builder.add_node("execute_tools",execute_tools)
builder.add_node("revise",revisor)

builder.add_edge("draft","execute_tools")
builder.add_edge("execute_tools","revise")

# Decide where to go next after revise node
def event_loop(state:List[BaseMessage])->str:
    count_tool_visits = sum(isinstance(item,ToolMessage) for item in state)
    if(count_tool_visits>MAX_ITERATIONS):
        return END
    return "execute_tools"

builder.add_conditional_edges("revise",event_loop)
builder.set_entry_point("draft")

graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path='graph.png')


if __name__=='__main__':
    print("Hello reflexion!")

    res = graph.invoke("Write about AI powered SOC/autonomous SOC domain, list the startups that do that and raised capital.")

    print(res)



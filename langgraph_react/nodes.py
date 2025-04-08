from dotenv import load_dotenv
load_dotenv()

from langgraph.prebuilt import ToolNode

from react import react_agent_runnable,tools
from state import AgentState

from langchain_core.messages import AIMessage

import uuid

from typing import Union, Dict, Any
from langchain_core.agents import AgentAction
from langchain_core.tools import BaseTool


def agent_action_to_tool_call(
    agent_action: AgentAction, tool_map: Dict[str, BaseTool]
) -> dict:
    """
    Convert an AgentAction to a tool_call dictionary that works with AIMessage(tool_calls=[...])
    in LangGraph ToolNode pipelines.
    Handles both string and dict forms of tool_input, avoiding assumptions about input key names.
    
    Parameters:
        agent_action (AgentAction): The action containing tool name and input.
        tool_map (Dict[str, BaseTool]): A dictionary mapping tool names to their tool objects.
    
    Returns:
        dict: A dictionary suitable for AIMessage(tool_calls=[...]).
    """
    tool_input = agent_action.tool_input
    tool_name = agent_action.tool

    if tool_name not in tool_map:
        raise ValueError(f"Tool '{tool_name}' not found in provided tool_map.")

    tool_schema = tool_map[tool_name].args_schema

    if isinstance(tool_input, dict):
        args = tool_input
    elif isinstance(tool_input, str):
        if tool_schema:
            fields = list(tool_schema.__fields__.keys())
            if len(fields) == 1:
                # Automatically assign the single field name to the string input
                args = {fields[0]: tool_input}
            else:
                raise ValueError(
                    f"Tool '{tool_name}' expects multiple inputs, but received a single string."
                )
        else:
            # If no schema is defined, fallback to 'input' key
            args = {"input": tool_input}
    else:
        raise ValueError(
            f"Unsupported type for tool_input: {type(tool_input)}. Must be str or dict."
        )

    return {
        "name": tool_name,
        "args": args,
        "id": str(uuid.uuid4()),
        "type": "tool_call"
    }

# This is the reasoning part(Re) in ReAct
def run_agent_reasoning_engine(state:AgentState):
    agent_outcome = react_agent_runnable.invoke(state)

    # This return basically updates the state(done by langgraph)
    return {"agent_outcome":agent_outcome}

tool_node = ToolNode(tools)
tool_map = {tool.name: tool for tool in tools}

# This is the Act part of ReAct
def execute_tools(state:AgentState):
    agent_action = state["agent_outcome"]

    ai_message = AIMessage(content="", tool_calls=[agent_action_to_tool_call(agent_action,tool_map)])
    output = tool_node.invoke([ai_message])

    return {"intermediate_steps":[(agent_action,str(output))]}



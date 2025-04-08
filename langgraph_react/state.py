import operator
from typing import TypedDict,Annotated, Union

from langchain_core.agents import AgentAction,AgentFinish


class AgentState(TypedDict):
    # This is the user input
    input:str
    # This is the output of the agent. It can be either AgentAction or AgentFinish. None is there bcoz when we first run a node, we might still not have any result yet.
    # Operator.add is not given to this attribute. So if the node exeuction result has an update for this attribute, its overwritten otherwise stays the same.
    agent_outcome :Union[AgentAction,AgentFinish,None]
    # This is the agent scratchpad part. It has to be AgentAction(otherwise we would have ended). The operator.add indicates langchain to append to this whenever a node returns this.
    intermediate_steps:Annotated[list[tuple[AgentAction,str]],operator.add]

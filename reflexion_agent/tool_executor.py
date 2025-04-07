from re import search
from typing import List
from dotenv import load_dotenv

import json

from schemas import AnswerQuestion, Reflection
from chains import parser
load_dotenv()

from langchain_core.messages import BaseMessage, ToolMessage,HumanMessage,AIMessage
from langgraph.prebuilt import ToolNode
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

from collections import defaultdict

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search,max_results=5)

tool_node = ToolNode([tavily_tool])

def execute_tools(state:List[BaseMessage])->List[ToolMessage]:
    tool_invocation:List[BaseMessage] = state[-1]
    parsed_tool_calls = parser.invoke(tool_invocation)

    ids = []
    tool_calls = []
    
    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            tool_calls.append(
                {
                "name":"tavily_search_results_json",
                "args":{"query":query},
                "id":parsed_call["id"],
                "type":"tool_call",
                }
            )
            ids.append(parsed_call["id"])

    ai_message = AIMessage(content="",tool_calls=tool_calls)

    outputs = tool_node.invoke([ai_message])

    outputs_map = defaultdict(dict)
    for id_,output,invocation in zip(ids,outputs,tool_calls):
        outputs_map[id_][invocation["args"]["query"]] = output.content
    
    tool_messages = []
    for id_,query_outputs in outputs_map.items():
        tool_messages.append(ToolMessage(content=json.dumps(query_outputs),tool_call_id=id_))
    
    return tool_messages


if __name__=='__main__':
    print("Tool executor")

    human_message = HumanMessage(
        content="Write about AI powered SOC/autonomous SOC domain, list the startups that do that and raised capital."
    )

    # This is just to test the tool executor. Observe that ID is returned by LLM and later used to correlated it with later
    answer = AnswerQuestion(answer="",reflection=Reflection(missing="",superfluous=""),
                            search_queries=["AI-powered SOC startups funding","AI SOC problem domain specifics","Technologies used by AI-powered SOC startups"],
                            id="call_kkkjiiij23",
                            )
    
    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name":AnswerQuestion.__name__,
                        "args":answer.dict(),
                        "id":"call_kkkjiiij23",
                    }
                ]
            )
        ]
    )

    print(raw_res)


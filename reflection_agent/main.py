from dotenv import load_dotenv

load_dotenv()

from typing import Sequence,List

from langchain_core.messages import BaseMessage, HumanMessage
# MessageGraph is a graph whose state is simply a Sequence of messages
from langgraph.graph import END,MessageGraph

from chains import generate_chain,reflect_chain

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state:Sequence[BaseMessage]):
    return generate_chain.invoke({"messages":state})

def reflection_node(state:Sequence[BaseMessage])->List[BaseMessage]:
    res = reflect_chain.invoke({"messages":state})
    # Change return message role from AI to human. This is done to trick LLM thinking reflection was done by human
    return [HumanMessage(content=res.content)]

builder = MessageGraph()
builder.add_node(GENERATE,generation_node)
builder.add_node(REFLECT,reflection_node)

builder.set_entry_point(GENERATE)

# This is the conditional edge which decides whether to end or continue with reflection over the generated response currently
def should_continue(state:List[BaseMessage]):
    if len(state)>6:
        return END
    return REFLECT

# This is to move from generate to reflect/end conditionally. This condition can be our own condition or can even use LLM. Anything can be there
builder.add_conditional_edges(GENERATE,should_continue)
# This is to move from reflect to generate
builder.add_edge(REFLECT,GENERATE)

graph = builder.compile()
# This output will generate a code which u can copy paste on mermaid to see visualization of graph
print(graph.get_graph().draw_mermaid())

if __name__=='__main__':
    print("Hello langgraph!")

    inputs = HumanMessage(content="""
    The following is my profile details:
    Machine Learning Researcher(M.S.ByResearch, IIT Madras)with deep learning,NLP,and computer vision expertise.
    Published at AAAI, skilled in adversarial robustness, interpretability, and transformer architectures.Developed AI
    models for image segmentation using PEFT,MoE,and hierarchical transfer learning with SegFormer.Built text
    summarization, semanticsearch, and transliteration tools using Transformers, LSTM, and attention mechanisms.
    Strong analytical and optimization skills, leveraging structured hyperparameter tuning and experiment tracking with
    Weights & Biases while fine-tuning,adapting and optimizing LLMs.Demonstrated technical writing expertise
    through research publications and extensive experience in analyzing research literature.Proven problem-solving
    skills as a former software engineer in a fast-paced startup, delivering scalable backend and frontend systems.
    """)

    response = graph.invoke(inputs)

    print(response)

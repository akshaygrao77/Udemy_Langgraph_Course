from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END, StateGraph

from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH


def decide_to_generate(state: GraphState):
    print("--------ACCESS GENERATION-----------")
    if state["web_search"] == True:
        print("--------DECIDED TO DO WEB SEARCH---------")
        return WEBSEARCH
    print("--------DECIDED TO DO GENERATION---------")
    return GENERATE


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve.retrieve)
workflow.add_node(GENERATE, generate.generate_answers)
workflow.add_node(WEBSEARCH, web_search.web_search)
workflow.add_node(GRADE_DOCUMENTS, grade_documents.grade_documents)

workflow.set_entry_point(RETRIEVE)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="condition_rag.png")

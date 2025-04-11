from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END, StateGraph

from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.chains import answer_grader, hallucination_grader
from graph.state import GraphState

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH


def decide_to_generate(state: GraphState):
    print("--------ACCESS GENERATION-----------")
    if state["web_search"] == True:
        print("--------DECIDED TO DO WEB SEARCH---------")
        return WEBSEARCH
    print("--------DECIDED TO DO GENERATION---------")
    return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState):
    print("----------- CHECK HALLUCINATIONS--------------")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.hallucination_grader.invoke(
        {"generation": generation, "documents": documents}
    )

    if hallucination_grade := score.binary_score:
        print("----DECISION: GENERATION IS GROUNDED IN DOCUMENTS----------")
        print("GRADE GENERATION VS QUESTION")
        score = answer_grader.answer_grader.invoke(
            {"question": question, "generation": generation}
        )

        if answer_grade := score.binary_score:
            print("--------- DECISION: GENERATION ADDRESSES QUESTION-------")
            # This will later be mapped to END node
            return "useful"
        else:
            print("--------- DECISION: GENERATION ADDRESSES QUESTION-------")
            # This will later go back to tavily search bcoz we didn't generate answer.
            # This is because the model didn't hallucinate and was grounded in the document. So this means, to get the answer we need the internet now.
            return "not_useful"
    else:
        print("----DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS----------")
        return "not_supported"


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

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {"not_useful": WEBSEARCH, "useful": END, "not_supported": GENERATE},
)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="condition_rag.png")

from dotenv import load_dotenv

load_dotenv()

from advanced_rag_langgraph.graph.chains.retrieval_grader import (
    GradeDocuments,
    retrieval_grader,
)
from advanced_rag_langgraph.ingestion import retriever

# ------------- Important -------------
# Testing LLM calls is very hard bcoz they are statistical, third-party so behaviour is not guaranteed.
#  Lastly it involves actually calling them and costs money!
# So, the below is a way to enforce testing ignoring above points.
# There are some advanced ways to bypass these issues but its very complex


# This checks if we are getting retriever grader gives YES answer when the relevant document is passed
def test_retreval_grader_answer_yes() -> None:
    # This is a question we know exists for sure
    question = "agent memory"

    docs = retriever.invoke(question)
    # The zeroth document retrieved will be the one with highest similarity score.
    # Since we know the question exists very much in our vector store, we know that this first document should definitely be relevant.
    doc_txt = docs[0].page_content

    # Check what the grade assigned by retriever grader for this obvious question
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


# This checks if we are getting retriever grader gives NO answer when the irrelevant document is passed
def test_retreval_grader_answer_no() -> None:
    # This is a question we know exists for sure
    question = "agent memory"

    docs = retriever.invoke(question)

    doc_txt = docs[0].page_content

    # Check what the grade assigned by retriever grader for this an irrelevant question by design
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "How to make a pizza", "document": doc_txt}
    )

    assert res.binary_score == "no"

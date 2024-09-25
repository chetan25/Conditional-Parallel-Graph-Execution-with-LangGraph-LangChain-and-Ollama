from typing import Any, Dict

from agent import (
    GraphState,
    generation_chain,
    document_grader_chain,
    answer_grader_chain,
    hallucination_grader_chain
)
from doc_ingestion import retriever

def retrieve(state:GraphState) -> Dict[str, Any]:
    """Retrieve documents form the source and update state"""
    print("===== retrieve: [Invoked] =====")

    question = state["question"]
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate Answer from the doc"""
    print("===== generate: [Invoked] =====")
    
    question = state["question"]
    documents = state["documents"]

    generations = generation_chain.invoke({
        "context": documents,
        "question": question
    })

    return {"documents": documents, "question": question, "answer": generations}


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determine whether retirved documents are relevant to the question
    If any document is not relevant we will filter it out
    Args:
        state(dict): The current graph state
    Returns:
        state(dict): Filtered out irrelevant docs and update web_search state   
    """

    print("===== grade_documents: [Invoked] =====")

    question = state['question']
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = document_grader_chain.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        print(f"the grade is {score}")
        if grade.lower() =="yes":
            print("===== grade_documents: [Document Relevant] =====")
            filtered_docs.append(d)
        else:
            print("===== grade_documents: [Document Not Relevant] =====")
            continue

    return { "documents": filtered_docs, "question": question}


def grade_generation_for_hallucination(state: GraphState) -> str:
    """Grade the final answer for hallucination"""
    print("===== grade_generation_for_hallucination: [Invoked] =====")

    documents = state["documents"]
    generations = state["answer"]

    score = hallucination_grader_chain.invoke(
        {"documents": documents, "generation": generations}
    )

    hallucination_grade = score.binary_score
    if hallucination_grade.lower() == "yes":
        print("===== grade_generation_for_hallucination: [Not Hallucinated] =====")
        return "not_hallucinated"
    else:
        print("===== grade_generation_for_hallucination: [Is Hallucinated] =====")
        return "hallucinated"

def grade_ans_for_hallucination(state: GraphState) -> Dict[str, Any]:
    """Grade the final answer for hallucination"""
    print("===== grade_generation_for_hallucination: [Invoked] =====")

    documents = state["documents"]
    generations = state["answer"]

    score = hallucination_grader_chain.invoke(
        {"documents": documents, "generation": generations}
    )

    hallucination_grade = score.binary_score
    if hallucination_grade.lower() == "yes":
        print("===== grade_generation_for_hallucination: [Not Hallucinated] =====")
        return { "isAnsHallucinated": 'No' }
    else:
        print("===== grade_generation_for_hallucination: [Is Hallucinated] =====")
        return { "isAnsHallucinated": 'Yes' }   


def grade_generation(state: GraphState) -> Dict[str, Any]:
    """Grade the final answer for relevance to question"""
    print("===== grade_generation: [Invoked] =====")

    question = state["question"]
    generations = state["answer"]
    documents = state["documents"]

    score = answer_grader_chain.invoke(
        {"question": question, "generation": generations}
    )

    answer_grade = score.binary_score
    if answer_grade.lower() == "yes":
        print("===== grade_generation: [Final Answer is relevant] =====")
        return { "question": question, "documents": documents, "answer":generations, "isAnsValid": 'Yes' }
    else:
        print("===== grade_generation: [Final Answer is not relevant] =====")
        return { "question": question, "documents": documents, "isAnsValid": 'No', "answer": "Sorry we cannot answer the question based on the documents we have" }
    
def check_ans(state: GraphState) -> Dict[str, Any]:
   isAnsHallucinated = state['isAnsHallucinated']
   isAnsValid = state['isAnsValid']
   answer: str = state['answer']

   if isAnsValid.lower() == 'yes' and isAnsHallucinated.lower() == 'no':
       return { "answer": answer }
   else:
       return { "answer": "Soory cannot answer the question with the provided documents"}
   

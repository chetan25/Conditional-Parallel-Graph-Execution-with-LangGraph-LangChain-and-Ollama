from typing import Any, Dict, Sequence
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent import (
    GraphState,
    retrieve,
    generate,
    grade_documents,
    grade_generation_for_hallucination,
    grade_generation
)


RETRIEVE = "retrieve"
GENERATE = "generate"
GRADE_DOC = "grade_doc"
GRADE_ANS = "grade_ans"
GRADE_ANS_HALLUCINATION = "ans_hallucination"
OUT_SCOPE_ANS = "out_Scope_ans"
HUMAN_FEEDBACK_AFTER_DOC_GRADE = "human_feedback_after_doc_grade"
MAX_ITERATION_RESPONSE= "max_iteration_response"


def should_generate_answer(state: GraphState) -> str:
    """Decide based on the length of the documents if we should genereate answer or not"""
    print("=====  should_generate_answer: [Invoked] ====")
    
    documents = state["documents"]

    if len(documents) == 0:
        print("===== should_generate_answer: [Decision No]  ====")
        return 'no'
    else:
        print("===== should_generate_answer: [Decision Yes]  ====")
        return 'yes'


def default_ans(state: GraphState) -> Dict[str, any]:
    """Return a default answer, statting question out of bound"""
    print("===== default_ans: [Invoked]  ====")
    
    documents = state["documents"]
    question = state["question"]

    return { "question": question, "documents": documents, "answer": "Sorry question out of scope"}


def human_feedback_after_doc_grade(state: GraphState) -> None:
    """Human in the loop step to stop the graph before this step to take feedback"""
    print("===== human_feedback_after_doc_grade: [Invoked] =====")


def max_iteration_response(state: GraphState) -> None:
    """Human in the loop step to stop the graph before this step to take feedback"""
    print("===== MAX_ITERATION_RESPONSE: [Invoked] =====")


def route_to_retrieveagain_or_end(state: GraphState) -> Sequence[str]:
    """Decide if we need to route to Genereate step or end based on human input"""
    print("===== route_to_retrieveagain_or_end: [Invoked] =====")

    if state["humanInput"].lower() == 'yes':
        print("===== route_to_retrieveagain_or_end: [Route to RETRIEVE] =====")
        return [RETRIEVE]
    print("===== route_to_retrieveagain_or_end: [Route to Out of Scope] =====")
    return [OUT_SCOPE_ANS]

def should_ask_for_human_input(state: GraphState) -> str:
    """Decide if we need to ask for human feedback based on documnets length"""
    print("===== should_ask_for_human_input: [Invoked] =====")
   
    documents = state['documents']
    retry_count = state['retryCount']

    if len(documents) == 0:
        if retry_count == 0:
            print("===== should_ask_for_human_input: [Asking for Human Input] =====")
            return HUMAN_FEEDBACK_AFTER_DOC_GRADE
        else:
            print("===== should_ask_for_human_input: [Informing Human of Retry Step result] =====")
            return MAX_ITERATION_RESPONSE
    else:
        print("===== should_ask_for_human_input: [Skipping Human Input and going to Generate] =====")
        return GENERATE

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GENERATE, generate)
workflow.add_node(GRADE_DOC, grade_documents)
workflow.add_node(GRADE_ANS, grade_generation)
# workflow.add_node(GRADE_ANS_HALLUCINATION, grade_generation_for_hallucination)
workflow.add_node(OUT_SCOPE_ANS, default_ans)
workflow.add_node(HUMAN_FEEDBACK_AFTER_DOC_GRADE, human_feedback_after_doc_grade)
workflow.add_node(MAX_ITERATION_RESPONSE, max_iteration_response)

workflow.set_entry_point(RETRIEVE)

workflow.add_edge(RETRIEVE, GRADE_DOC)

workflow.add_conditional_edges(
    GRADE_DOC,
    should_ask_for_human_input,
    {
        HUMAN_FEEDBACK_AFTER_DOC_GRADE: HUMAN_FEEDBACK_AFTER_DOC_GRADE,
        GENERATE: GENERATE,
        MAX_ITERATION_RESPONSE: MAX_ITERATION_RESPONSE
    },
)


# all nodes where the graph can divert to
intermediates = [OUT_SCOPE_ANS, RETRIEVE]

workflow.add_conditional_edges(
    HUMAN_FEEDBACK_AFTER_DOC_GRADE,
    route_to_retrieveagain_or_end,
   intermediates
)

# This is either or flow
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_for_hallucination,
    {
        'not_hallucinated': GRADE_ANS,
        'hallucinated': OUT_SCOPE_ANS
    }
)

# workflow.add_edge(HUMAN_FEEDBACK_AFTER_DOC_GRADE, RETRIEVE)
workflow.add_edge(MAX_ITERATION_RESPONSE, OUT_SCOPE_ANS)
workflow.add_edge(GRADE_ANS, END)
workflow.add_edge(OUT_SCOPE_ANS, END)

memory = MemorySaver()

# we tell the graph to stop at this step
graph_sequence = workflow.compile(checkpointer=memory, interrupt_before=[HUMAN_FEEDBACK_AFTER_DOC_GRADE, MAX_ITERATION_RESPONSE])
graph_sequence.get_graph().draw_mermaid_png(output_file_path='./graph-human-in-loop.png')
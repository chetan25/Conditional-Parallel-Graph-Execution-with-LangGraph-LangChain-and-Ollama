from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of the graph

    Attributes:
        question: question
        answer: LLM answer
        documents: List of documents
        humanInput: Human input to continue or not
        retryCount: Count the number of retry based on human input
        isAnsHallucinated: Binary score of yes or no for is final answer has hallucination or not
        isAnsValid: Binary score of yes or no for if the final answer, answers users question
    """

    question: str
    answer: str
    documents: List[str]
    humanInput: str
    retryCount: int
    isAnsHallucinated: str
    isAnsValid: str
    
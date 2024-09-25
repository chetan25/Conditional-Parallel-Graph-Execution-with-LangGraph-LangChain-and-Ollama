from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_ollama import ChatOllama
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from prompts import (
    grader_prompt,
    answer_grader_prompt,
    hallucination_prompt,
    GradeDocuments,
    GradeHallucination,
    GradeAnswer
)

# LOCAL_LLM = "llama3-groq-tool-use"
LOCAL_LLM = "llama3.1"
llm = OllamaFunctions(model=LOCAL_LLM, format="json", temperature=0)
llm2 = OllamaFunctions(model="llama3-groq-tool-use", format="json", temperature=0)

# llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)
# llm2 = ChatOllama(model="llama3-groq-tool-use", format="json", temperature=0)

generation_prompt = hub.pull("rlm/rag-prompt")

generation_chain = generation_prompt | llm  | StrOutputParser()

# use function calling will return pydantic object in schema we want
# to use with_structured_output we need to use model that supports function calling
structured_document_grader_llm = llm2.with_structured_output(GradeDocuments)
document_grader_chain = grader_prompt | structured_document_grader_llm


structured_hallucination_grader_llm = llm.with_structured_output(GradeHallucination)
hallucination_grader_chain = hallucination_prompt | structured_hallucination_grader_llm


structured__answer_grader_llm = llm.with_structured_output(GradeAnswer)
answer_grader_chain = answer_grader_prompt | structured__answer_grader_llm
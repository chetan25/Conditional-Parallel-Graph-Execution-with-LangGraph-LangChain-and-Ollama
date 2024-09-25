from dotenv import load_dotenv
load_dotenv()
from doc_ingestion import vectorstore
from agent import graph_sequence 
from agent import graph_parallel


if __name__ == "__main__":
    # graph = graph_sequence
    graph = graph_parallel
    retry_count = 0
    MAX_RETRY_COUNT = 1
    # results = vectorstore.similarity_search(
    #     query="How many layers does Earth have"
    # )
    thread = {"configurable": {"thread_id": "2222"}}

    initial_input = {"question": "How many layers does Earth has", "retryCount": 0}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        # print(event)
        pass
    
    if len(graph.get_state(thread).next) > 1:

        # print(len(graph.get_state(thread).next), "length")
        print("============= HUMAN INPUT ====================")
        user_input = input("================= \n The Question cannot be answered based on the documents provides, Do you wish to try Retrieving step again, 'yes' or 'no' ?")
        retry_count += 1
        graph.update_state(thread, {"humanInput": user_input, "documents": [], "retryCount": retry_count}, as_node="human_feedback_after_doc_grade")
    
        # print("--State after update--")
        # print(graph.get_state(thread))

        # print(graph.get_state(thread).next)
        print("============= HUMAN INPUT AGAIN====================")
        user_interrupt = input("================= \n Sorry we cannot reterieve any documents relevant to the question. Press a key to continue")
        # this way we prevent going more than the max allowed
        graph.update_state(thread, {"humanInput": 'no'}, as_node="max_iteration_response")    
    
        for event in graph.stream(None, thread, stream_mode="values"):
            # print(event)
            pass

     
    print("============= FINAL ANSWER ====================")
    print(graph.get_state(thread).values["answer"])

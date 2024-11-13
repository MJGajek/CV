# interface.py


import gradio as gr
from retriever import retrieve_top_chunks
from generator import generate_response

def handle_query(query, index, metadata, retriever_model, model, tokenizer, streamer, retriever_params, generator_params):
    """Handles a query by retrieving context and generating a response."""
    # Retrieve context as a list of dictionaries
    retrieved_chunks = retrieve_top_chunks(query, index, metadata, retriever_model, **retriever_params)

    # Concatenate text fields from retrieved chunks
    context = "\n".join(chunk["speaker"] + ":" + "\n" + chunk["text"] for chunk in retrieved_chunks)

    # Generate response using concatenated context and generator_params
    response = generate_response(query, context, model, tokenizer, streamer, **generator_params)

    return response

def gradio_interface(index, metadata, retriever_model, model, tokenizer, streamer, retriever_params, generator_params):
    """Sets up and launches the Gradio interface for the chatbot."""

    def gradio_handle_query(query):
        """Wrapper function to handle Gradio inputs."""
        return handle_query(query, index, metadata, retriever_model, model, tokenizer, streamer, retriever_params, generator_params)

    # Define Gradio interface
    gradio_app = gr.Interface(
        fn=gradio_handle_query,
        inputs="text",
        outputs="text",
        title="Conference Q&A Chatbot",
        description="Ask questions about the conference. The chatbot will retrieve context and provide an answer."
    )

    # Launch the Gradio interface
    gradio_app.launch()
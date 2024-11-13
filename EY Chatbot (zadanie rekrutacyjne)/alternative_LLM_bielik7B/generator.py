# generator.py

import torch
import logging
from transformers import AutoTokenizer, BitsAndBytesConfig, TextStreamer
from vllm import LLM, SamplingParams

def initialize_generator(model_name="speakleash/Bielik-7B-Instruct-v0.1-AWQ"):
    """Loads the Bielik AWQ quantized model and tokenizer with adjusted settings to reduce memory usage."""
    logging.info("Loading Bielik AWQ quantized model and tokenizer...")
    try:
        sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=256)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        model = LLM(
            model=model_name,
            quantization='awq',
            dtype='half',
            gpu_memory_utilization=0.5,  # Further reduce memory utilization
            max_model_len=1500,          # Reduce max tokens
            enforce_eager=True,          # Force eager mode to save memory
            enable_prefix_caching=False,  # Disable caching to reduce memory
            device=device_type            # Explicitly set device type
        )

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        logging.info("Bielik AWQ model and tokenizer loaded successfully.")
        
        return model, tokenizer, streamer, sampling_params
    except Exception as e:
        logging.error("Failed to load Bielik AWQ model and tokenizer. Error:", exc_info=True)
        return None, None, None, None


def generate_response(query, context, model, tokenizer, streamer, sampling_params):
    """Generates a response using the Bielik model, with debugging output to check generation."""
    # Simplified prompt for debugging
    system_prompt = (
    "Twoim zadaniem jest odpowiedzieć na pytanie korzystając tylko z fragmentów kontekstu z konferencj, które dostaniesz."
    "Powiedz kto jest autorem fragmentu z kontekstu"
    )
    user_prompt = f"KONTEKST:\n{context}\nPYTANIE:\n{query}"
    prompt = f"{system_prompt}\n{user_prompt}"
    
    # Print the prompt to verify its structure
    #print("Prompt for Generation:", prompt)

    # Generate the response with debugging for the output structure
    output = model.generate(prompt, sampling_params)
    
    # Print the raw output structure for debugging
    #print("Raw Model Output:", output)

    # Check if the output is non-empty and properly structured
    if output and hasattr(output[0], "outputs") and output[0].outputs:
        response = output[0].outputs[0].text.strip()
        #print("Generated Response:", response)
    else:
        print("No output generated.")
        response = "No response generated."

    return response

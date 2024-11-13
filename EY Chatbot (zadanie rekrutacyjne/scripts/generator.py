# generator.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import logging

def initialize_generator(model_name="eryk-mazus/polka-1.1b-chat"):
    """Loads the generator model and tokenizer."""
    logging.info("Loading generator model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        logging.info("Generator model and tokenizer loaded successfully.")
        return model, tokenizer, streamer
    except Exception as e:
        logging.error("Failed to load generator model and tokenizer.", exc_info=True)
        return None, None, None
        
def generate_response(query, context, model, tokenizer, streamer, **generation_params):
    """Generates a response using the generator model, based on query and retrieved context."""
    system_prompt = (
    "Twoim zadaniem jest odpowiedzieć na pytanie korzystając tylko z fragmentów kontekstu z konferencj, które dostaniesz."
    "Powiedz kto jest autorem fragmentu z kontekstu"
    )
    chat = [{"role": "system", "content": system_prompt}]
    user_prompt = f"KONTEKST:\n{context}\nPYTANIE:\n{query}"
    chat.append({"role": "user", "content": user_prompt})

    # Add assistant context for continuity if previous responses are relevant to the current query
    #chat.append({"role": "assistant", "content": context})

    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)
    print(chat)

    # Generate the response with the specified parameters
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            pad_token_id=tokenizer.eos_token_id,
            **generation_params,  # Pass parameters dynamically
            streamer=streamer,
        )

    # Extract and decode the new tokens
    new_tokens = outputs[0, inputs.size(1):]
    response = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Custom post-processing to remove all special tokens except <eos>
    special_tokens_to_remove = [token for token in tokenizer.all_special_tokens if token != tokenizer.eos_token]
    for token in special_tokens_to_remove:
        response = response.replace(token, "")

    logging.info(f"Generated response length: {len(response)} characters.")
    return response


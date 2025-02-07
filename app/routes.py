# app/routes.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import torch
from fastapi.templating import Jinja2Templates
import logging
from datetime import datetime

from app.models import model, tokenizer, SYSTEM_PROMPTS, SYSTEM_PROMPT, load_model  # note the import of SYSTEM_PROMPTS
from app.schemas import GenerationRequest, ChangeModelRequest

router = APIRouter()
templates = Jinja2Templates(directory="templates")

MODEL_MAP = {
    "de0b4c67-de5e-4bbf-82ec-6fffce8e1b83": "meta-llama/Llama-3.1-8B",
    "3cb9bc3f-05a8-4644-8b83-9d7010edf301": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
}


BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SYSTEM_ROLE = "<system>"
SYSTEM_ROLE_END = "</system>"
USER_ROLE = "<user>"
USER_ROLE_END = "</user>"

@router.post("/generate")
def generate_text(request_data: GenerationRequest, request: Request):
    if request_data.max_new_tokens > 200:
        request_data.max_new_tokens = 200

    client_ip = request.client.host if request.client else "unknown"
    timestamp = datetime.utcnow().isoformat()

    # Select the system prompt based on the provided choice.
    selected_system_prompt = SYSTEM_PROMPTS.get(request_data.system_prompt_choice, SYSTEM_PROMPT)
    
    logging.info(
        f"Generation request from {client_ip} at {timestamp} with payload: {request_data.json()}"
    )

    final_prompt = (
        f"{BOS_TOKEN}"
        f"{SYSTEM_ROLE}{selected_system_prompt}{SYSTEM_ROLE_END}\n"
        f"{USER_ROLE}{request_data.user_prompt}{USER_ROLE_END}\n"
        "<assistant>"
    )

    prompt_inputs = tokenizer(final_prompt, return_tensors="pt", padding=True)
    input_ids = prompt_inputs["input_ids"].to(model.device)
    attention_mask = prompt_inputs["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    user_inputs = tokenizer(request_data.user_prompt, return_tensors="pt")
    user_token_ids = user_inputs["input_ids"][0]
    user_tokens = [tokenizer.decode([tid]) for tid in user_token_ids]

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=request_data.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )

    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    output_token_ids = generated_ids[0][prompt_len:]
    generated_text_only = tokenizer.decode(output_token_ids, skip_special_tokens=True)
    
    if "</assistant>" in generated_text_only:
        generated_text_only = generated_text_only.split("</assistant>")[0]
    
    output_tokens = tokenizer(generated_text_only, return_tensors="pt")["input_ids"]
    output_tokens = [tokenizer.decode([tid]) for tid in output_tokens[0]]

    logging.info(
        f"Generated output for {client_ip} at {timestamp}: {generated_text_only}"
    )

    return {
        "system_prompt": selected_system_prompt,
        "combined_prompt": final_prompt,
        "user_tokens": user_tokens,
        "generated_text_only": generated_text_only,
        "output_tokens": output_tokens,
    }

@router.get("/get_prompt")
def get_prompt(key: str):
    # Assume SYSTEM_PROMPTS is your dictionary of prompts
    prompt_text = SYSTEM_PROMPTS.get(key, "Prompt not found.")
    return {"prompt_text": prompt_text}

@router.post("/change_model")
def change_model(request_data: ChangeModelRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    timestamp = datetime.utcnow().isoformat()

    logging.info(
        f"Change model request from {client_ip} at {timestamp} with payload: {request_data.json()}"
    )

    # Look up the model name from the UUID provided.
    model_uuid = request_data.model_uuid
    if model_uuid not in MODEL_MAP:
        error_msg = f"Invalid model UUID: {model_uuid}"
        logging.error(error_msg)
        return {"status": "error", "error": error_msg}

    model_name = MODEL_MAP[model_uuid]

    try:
        load_model(model_name)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return {"status": "error", "error": str(e)}

    logging.info(f"Model changed successfully to {model_name}")
    return {"status": "success", "model_name": model_name}

@router.get("/", response_class=HTMLResponse)
def root(request: Request):
    # Pass the available system prompt options to the template so that the front end can render a toggle.
    prompt_options = list(SYSTEM_PROMPTS.keys())
    return templates.TemplateResponse("index.html", {"request": request, "system_prompt": SYSTEM_PROMPT, "prompt_options": prompt_options})

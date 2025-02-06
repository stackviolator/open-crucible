# routes.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import torch
from fastapi.templating import Jinja2Templates
import logging
from datetime import datetime

from app.models import model, tokenizer, SYSTEM_PROMPT
from app.schemas import GenerationRequest

router = APIRouter()
templates = Jinja2Templates(directory="templates")

BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SYSTEM_ROLE = "<system>"
SYSTEM_ROLE_END = "</system>"
USER_ROLE = "<user>"
USER_ROLE_END = "</user>"

@router.post("/generate")
def generate_text(request_data: GenerationRequest, request: Request):
    # Cap max_new_tokens to 200 if it's higher
    if request_data.max_new_tokens > 200:
        request_data.max_new_tokens = 200

    # Retrieve the client IP address and current timestamp
    client_ip = request.client.host if request.client else "unknown"
    timestamp = datetime.utcnow().isoformat()

    # Log the incoming generation request with details
    logging.info(
        f"Generation request from {client_ip} at {timestamp} with payload: {request_data.json()}"
    )

    final_prompt = (
        f"{BOS_TOKEN}"
        f"{SYSTEM_ROLE}{SYSTEM_PROMPT}{SYSTEM_ROLE_END}\n"
        f"{USER_ROLE}{request_data.user_prompt}{USER_ROLE_END}\n"
        "<assistant>"
    )

    # Tokenize with padding=True and retrieve attention_mask
    prompt_inputs = tokenizer(
        final_prompt,
        return_tensors="pt",
        padding=True
    )
    input_ids = prompt_inputs["input_ids"].to(model.device)
    attention_mask = prompt_inputs["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    # For display, tokenize user prompt alone
    user_inputs = tokenizer(request_data.user_prompt, return_tensors="pt")
    user_token_ids = user_inputs["input_ids"][0]
    user_tokens = [tokenizer.decode([tid]) for tid in user_token_ids]

    # Generate text, explicitly setting pad_token_id
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

    # Decode the entire output WITHOUT skipping special tokens
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    # The newly generated portion is everything beyond the original prompt length
    output_token_ids = generated_ids[0][prompt_len:]
    generated_text_only = tokenizer.decode(output_token_ids, skip_special_tokens=True)
    
    # Truncate at </assistant> if present in the trimmed portion (optional)
    if "</assistant>" in generated_text_only:
        generated_text_only = generated_text_only.split("</assistant>")[0]
    
    # Generate output tokens from the cleaned generated_text_only
    output_tokens = tokenizer(generated_text_only, return_tensors="pt")["input_ids"]
    output_tokens = [tokenizer.decode([tid]) for tid in output_tokens[0]]

    # Log the full generated output (full_text) along with the generation details
    logging.info(
        f"Generated output for {client_ip} at {timestamp}: {full_text}"
    )

    return {
        "system_prompt": SYSTEM_PROMPT,
        "combined_prompt": final_prompt,
        "user_tokens": user_tokens,
        "generated_text_only": generated_text_only,
        "output_tokens": output_tokens,
    }

@router.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "system_prompt": SYSTEM_PROMPT})

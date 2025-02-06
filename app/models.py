# models.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.1-8B"
CUSTOM_PAD_TOKEN = "[PAD]"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": CUSTOM_PAD_TOKEN})
    print(f"Added pad token: {CUSTOM_PAD_TOKEN}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Save current device (e.g., cuda:0 if available)
device = next(model.parameters()).device

# Move model to CPU to perform the resize, which can be less memory-intensive on CPU
model = model.to("cpu")
print("Resizing token embeddings on CPU...")
model.resize_token_embeddings(len(tokenizer))
print("Resizing complete.")

# Move the model back to GPU (or its original device)
model = model.to(device)
print("Model moved back to", device)

# app/models.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

CUSTOM_PAD_TOKEN = "[PAD]"

# Function to load levels from a YAML file.
def load_levels_from_yaml(file_path="app/levels.yaml"):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    # Convert list of levels to a dictionary keyed by level name for easy access.
    levels = {level["name"]: level for level in data.get("levels", [])}
    return levels

LEVELS = load_levels_from_yaml(os.getenv("LEVELS_FILE"))

model = None
tokenizer = None

def load_model(model_name: str):
    global model, tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding side (optional but often useful)
    tokenizer.padding_side = "right"

    # If the tokenizer doesn't have a dedicated pad token, add one.
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": CUSTOM_PAD_TOKEN})
        print(f"Added pad token: {CUSTOM_PAD_TOKEN}")

    # Explicitly ensure pad_token is set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = CUSTOM_PAD_TOKEN
    # Ensure pad_token_id is set; if not, fall back to eos_token_id.
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
        vocab = tokenizer.get_vocab()
        # Use the pad token's id if available, otherwise use eos_token_id.
        tokenizer.pad_token_id = vocab.get(CUSTOM_PAD_TOKEN, tokenizer.eos_token_id)
        print(f"Set pad token ID to: {tokenizer.pad_token_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Update the model's config with the pad token id.
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Resizing token embeddings on CPU...")
    if model.device.type == "mps":
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    else:
        model.resize_token_embeddings(len(tokenizer))
    print("Resizing complete.")
    print(f"Pad token is now: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    return model

# My CUDA machine is itty bitty so I use the 4-bit quantized model. 
# My mac is bigger so then I use the 16-bit model.
if torch.cuda.is_available():
    DEFAULT_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
else:
    DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B"

print("Loading default model...")
load_model(DEFAULT_MODEL_NAME)
print("Default model loaded.")

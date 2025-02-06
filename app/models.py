import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.1-8B"
CUSTOM_PAD_TOKEN = "[PAD]"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": CUSTOM_PAD_TOKEN})
    print(f"Added pad token: {CUSTOM_PAD_TOKEN}")

print("Loading model on CPU...")
# Force the model to load on CPU by setting device_map="cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu"
)

print("Resizing token embeddings on CPU...")
model.resize_token_embeddings(len(tokenizer))
print("Resizing complete.")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

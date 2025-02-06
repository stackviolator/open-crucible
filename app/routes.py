# routes.py

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
import torch

from models import model, tokenizer, SYSTEM_PROMPT
from schemas import GenerationRequest

router = APIRouter()

BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SYSTEM_ROLE = "<system>"
SYSTEM_ROLE_END = "</system>"
USER_ROLE = "<user>"
USER_ROLE_END = "</user>"

@router.post("/generate")
def generate_text(request: GenerationRequest):
    """
    Construct a prompt of:
      <BOS><system>SYSTEM_PROMPT</system><user>user_prompt</user><EOS>

    Generate text, then return it. Print the generated text to console.
    """
    final_prompt = (
        f"{BOS_TOKEN}"
        f"{SYSTEM_ROLE}{SYSTEM_PROMPT}{SYSTEM_ROLE_END}\n"
        f"{USER_ROLE}{request.user_prompt}{USER_ROLE_END}\n"
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
    user_inputs = tokenizer(request.user_prompt, return_tensors="pt")
    user_token_ids = user_inputs["input_ids"][0]
    user_tokens = [tokenizer.decode([tid]) for tid in user_token_ids]

    # Generate text, explicitly setting pad_token_id
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,       # Pass the attention mask
            max_new_tokens=request.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )

    # Decode the entire output WITHOUT skipping special tokens,
    # just to see everything. You can switch back later if you want.
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    # The newly generated portion is everything beyond the original prompt length
    output_token_ids = generated_ids[0][prompt_len:]
    # If you do want to skip special tokens for the "new" text:
    generated_text_only = tokenizer.decode(output_token_ids, skip_special_tokens=True)
    
    # Truncate at </assistant> if present
    if "</assistant>" in generated_text_only:
        generated_text_only = generated_text_only.split("</assistant>")[0]
    
    # Generate output tokens from the cleaned generated_text_only instead
    output_tokens = tokenizer(generated_text_only, return_tensors="pt")["input_ids"]
    output_tokens = [tokenizer.decode([tid]) for tid in output_tokens[0]]

    # Print to console
    print("----- Generated Text (Raw) -----")
    print(full_text)
    print("--------------------------------")

    return {
        "system_prompt": SYSTEM_PROMPT,
        "combined_prompt": final_prompt,
        "user_tokens": user_tokens,
        "generated_text_only": generated_text_only,
        "output_tokens": output_tokens,
    }

@router.get("/", response_class=HTMLResponse)
def root():
    # Same HTML you had before...
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Open Crucible</title>
    </head>
    <body style="font-family: sans-serif;">
        <h1>We <3 Open Crucible</h1>
        
        <h2>System Prompt:</h2>
        <div style="border:1px solid #CCC; padding:10px; background-color:#F9F9F9;">
            {SYSTEM_PROMPT}
        </div>
        
        <h2>User Prompt:</h2>
        <div style="border:1px solid #CCC; padding:10px; background-color:#F9F9F9;">
            <textarea id="userPrompt" rows="4" cols="70" placeholder="Type your prompt here..."></textarea><br/><br/>
        </div>
        
        <label for="maxTokens">Max new tokens:</label>
        <input id="maxTokens" type="number" value="100" /><br/><br/>
        
        <button onclick="sendPrompt()">Submit</button>
        <hr />

        <h2>Prompt Passed to Model:</h2>
        <div id="displayCombinedPrompt" style="border:1px solid #CCC; padding:10px; min-height:40px;"></div>

        <h2>User Tokens:</h2>
        <div id="displayUserTokens" style="border:1px solid #CCC; padding:10px; min-height:40px;"></div>
        
        <h2>Model Output:</h2>
        <div id="displayOutputText" style="border:1px solid #CCC; padding:10px; min-height:40px;"></div>

        <h2>Output Tokens:</h2>
        <div id="displayOutputTokens" style="border:1px solid #CCC; padding:10px; min-height:40px;"></div>

        <script>
        async function sendPrompt() {{
            const userPrompt = document.getElementById("userPrompt").value;
            const maxTokens = parseInt(document.getElementById("maxTokens").value) || 100;

            const payload = {{
                user_prompt: userPrompt,
                max_new_tokens: maxTokens
            }};

            document.getElementById("displayCombinedPrompt").innerText = "";
            document.getElementById("displayUserTokens").innerText = "";
            document.getElementById("displayOutputText").innerText = "";
            document.getElementById("displayOutputTokens").innerText = "";

            try {{
                const response = await fetch("/generate", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify(payload)
                }});
                const data = await response.json();

                document.getElementById("displayCombinedPrompt").innerText = data.combined_prompt;
                document.getElementById("displayUserTokens").innerText = JSON.stringify(data.user_tokens, null, 2);
                document.getElementById("displayOutputText").innerText = data.generated_text_only;
                document.getElementById("displayOutputTokens").innerText = JSON.stringify(data.output_tokens, null, 2);

            }} catch (error) {{
                console.error("Error:", error);
                document.getElementById("displayOutputText").innerText = "An error occurred: " + error;
            }}
        }}
        </script>
    </body>
    </html>
    """
    return html_content

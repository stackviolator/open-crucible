# app/routes.py

from fastapi import Request, Response, Depends, APIRouter, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from app.models import model, tokenizer, SYSTEM_PROMPTS, SYSTEM_PROMPT, load_model  # note the import of SYSTEM_PROMPTS
from app.schemas import GenerationRequest, ChangeModelRequest, RegisterRequest, LoginRequest, UpdateLevelRequest
from app.auth import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, set_auth_cookie, verify_password, get_password_hash
from app.dependencies import get_session
from app.db import engine, SessionLocal
from app.user_models import Base, User

from authlib.integrations.starlette_client import OAuth, OAuthError
from passlib.context import CryptContext

from datetime import datetime
import os
import logging
import torch
import uuid

router = APIRouter()
templates = Jinja2Templates(directory="templates")

MODEL_MAP = {
    "de0b4c67-de5e-4bbf-82ec-6fffce8e1b83": "meta-llama/Llama-3.1-8B",
    "3cb9bc3f-05a8-4644-8b83-9d7010edf301": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
}

# Special tokens
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SYSTEM_ROLE = "<system>"
SYSTEM_ROLE_END = "</system>"
USER_ROLE = "<user>"
USER_ROLE_END = "</user>"
ASSISTANT_ROLE = "<assistant>"
ASSISTANT_ROLE_END = "</assistant>"

# Add at the top of the file with other imports
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

def strip_special_tokens(text: str) -> str:
    # Build the list from the defined constants instead of hard coding
    tokens = [SYSTEM_ROLE, SYSTEM_ROLE_END, USER_ROLE, USER_ROLE_END, ASSISTANT_ROLE, ASSISTANT_ROLE_END]
    for token in tokens:
        text = text.replace(token, "")
    return text.strip()


# --- New ChatHistory Model ---
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    conversation_id = Column(String, index=True)  # New column to tie records to a conversation UUID
    user_prompt = Column(String)       # Will store the user prompt or system prompt (with tokens)
    assistant_reply = Column(String)   # Will store the assistant reply (with tokens)
    timestamp = Column(String)         # stored as ISO string


# Create the tables if they don't exist.
Base.metadata.create_all(bind=engine)
print("Tables created:", Base.metadata.tables.keys())

# Dependency to get a database session.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Generate Text ---
@router.post("/generate")
def generate_text(
    request_data: GenerationRequest,
    request: Request,
    response: Response,
    session: dict = Depends(get_session),
    db: Session = Depends(get_db)
):
    if isinstance(session, Response):
        return session
    """
    Generate text using the language model based on the user's prompt.
    Records the conversation (system prompt, user prompt, and assistant reply) in the database,
    tied to a conversation UUID.
    """
    # Check if the requested prompt level is accessible
    if int(request_data.system_prompt_choice.split('-')[1]) > session.get("highest_level", 0):
        raise HTTPException(status_code=403, detail="You do not have access to this prompt.")
    
    # Limit new tokens
    if request_data.max_new_tokens > 200:
        request_data.max_new_tokens = 200

    timestamp = datetime.utcnow().isoformat()
    selected_system_prompt = SYSTEM_PROMPTS.get(request_data.system_prompt_choice, SYSTEM_PROMPT)
    
    if DEBUG_MODE:
        print(f"DEBUG: Timestamp: {timestamp}")
        print(f"DEBUG: Selected system prompt: {selected_system_prompt}")

    # --- Conversation UUID Management ---
    conversation_reset = session.get("reset_conversation", False)
    conversation_id = session.get("conversation_id")
    if conversation_reset or not conversation_id:
        conversation_id = str(uuid.uuid4())
        session["conversation_id"] = conversation_id
        if "reset_conversation" in session:
            session.pop("reset_conversation")
        system_record = ChatHistory(
            username=session["sub"],
            conversation_id=conversation_id,
            user_prompt=f"<system>{selected_system_prompt}</system>",
            assistant_reply="",
            timestamp=timestamp
        )
        db.add(system_record)
        db.commit()
        if DEBUG_MODE:
            print(f"DEBUG: New conversation started. conversation_id set to: {conversation_id}")

    # --- Load Existing Conversation History from DB ---
    conversation_records = (
        db.query(ChatHistory)
          .filter(ChatHistory.username == session["sub"], ChatHistory.conversation_id == conversation_id)
          .order_by(ChatHistory.id.asc())
          .all()
    )
    if DEBUG_MODE:
        print(f"DEBUG: Loaded {len(conversation_records)} records for conversation_id {conversation_id}")

    conversation_text = ""
    for record in conversation_records:
        if record.user_prompt:
            conversation_text += record.user_prompt + "\n"
        if record.assistant_reply:
            conversation_text += record.assistant_reply + "\n"
    
    if DEBUG_MODE:
        print("DEBUG: Constructed conversation text:\n", conversation_text)

    # Build final prompt
    conversation_text += f"<user>{request_data.user_prompt}</user>\n<assistant>"
    final_prompt = f"{BOS_TOKEN}\n{conversation_text}"
    if DEBUG_MODE:
        print("DEBUG: Final prompt sent to model:\n", final_prompt)

    # --- Generate Model Output ---
    prompt_inputs = tokenizer(final_prompt, return_tensors="pt", padding=True)
    input_ids = prompt_inputs["input_ids"].to(model.device)
    attention_mask = prompt_inputs["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    # Get user input tokens for visualization
    user_input_tokens = [tokenizer.decode(token) for token in input_ids[0]]
    
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
    
    # Get output tokens for visualization
    output_token_ids = generated_ids[0][prompt_len:]
    output_tokens = [tokenizer.decode(token) for token in output_token_ids]
    
    generated_text_only = tokenizer.decode(output_token_ids, skip_special_tokens=True)
    if "</assistant>" in generated_text_only:
        generated_text_only = generated_text_only.split("</assistant>")[0]

    # Re-encode generated_text_only to get tokens for visualization
    output_tokens_ids = tokenizer.encode(generated_text_only)
    output_tokens = [tokenizer.decode(token) for token in output_tokens_ids]

    if DEBUG_MODE:
        print("DEBUG: Generated text only:", generated_text_only)

    # --- Jailbreak Check and Level Update ---
    lower_text = generated_text_only.lower()
    jailbreak_success = "one" in lower_text and "morbillion" in lower_text

    try:
        prompt_level = int(request_data.system_prompt_choice.split('-')[1])
        if jailbreak_success and prompt_level >= session.get("highest_level", 0):
            new_level = session.get("highest_level", 1) + 1
            user = db.query(User).filter(User.username == session["sub"]).first()
            if user:
                user.highest_level = new_level
                db.commit()
                token_data = {"sub": user.username, "highest_level": new_level}
                session.update(token_data)
                set_auth_cookie(response, token_data)
            if DEBUG_MODE:
                print(f"DEBUG: Jailbreak success detected. Leveling up to {new_level}")
            
            # Reset conversation after level up
            conversation_id = str(uuid.uuid4())
            session["conversation_id"] = conversation_id
            system_record = ChatHistory(
                username=session["sub"],
                conversation_id=conversation_id,
                user_prompt=f"<system>{selected_system_prompt}</system>",
                assistant_reply="",
                timestamp=timestamp
            )
            db.add(system_record)
            db.commit()
            if DEBUG_MODE:
                print("DEBUG: Conversation reset after level up. New conversation_id:", conversation_id)
    except (AttributeError, IndexError, ValueError) as e:
        logging.warning(f"Error processing level update: {str(e)}. system_prompt_choice: {request_data.system_prompt_choice}, session: {session}")

    # Store the conversation turn
    new_record = ChatHistory(
        username=session["sub"],
        conversation_id=conversation_id,
        user_prompt=f"<user>{request_data.user_prompt}</user>",
        assistant_reply=f"<assistant>{generated_text_only}</assistant>",
        timestamp=timestamp
    )
    db.add(new_record)
    db.commit()

    if DEBUG_MODE:
        print("DEBUG: New conversation turn stored for conversation_id:", conversation_id)

    return {
        "system_prompt": session.get("highest_level"),
        "combined_prompt": final_prompt,
        "generated_text_only": generated_text_only,
        "jailbreak_success": jailbreak_success,
        "user_tokens": user_input_tokens,  # Add token information
        "output_tokens": output_tokens,    # Add token information
    }

@router.get("/get_prompt")
def get_prompt(key: int, session: dict = Depends(get_session)):
    if isinstance(session, Response):
        return session
    """
    Retrieve the system prompt corresponding to the specified key.
    """
    level = session.get("highest_level")
    if key > level:
        raise HTTPException(status_code=403, detail="You do not have access to this prompt.")

    max_level = max(int(k.split('-')[1]) for k in SYSTEM_PROMPTS.keys())
    actual_key = min(key, max_level)
    prompt_text = SYSTEM_PROMPTS.get(f"level-{actual_key}", "Prompt not found.")
    return {"prompt_text": prompt_text}

@router.post("/change_model")
def change_model(
    request_data: ChangeModelRequest,
    request: Request,
    session: dict = Depends(get_session)
):
    if isinstance(session, Response):
        return session
    """
    Change the currently active language model.
    """
    if not session:
        raise HTTPException(status_code=403, detail="Invalid session")

    client_ip = request.client.host if request.client else "unknown"
    timestamp = datetime.utcnow().isoformat()

    logging.info(
        f"Change model request from {client_ip} at {timestamp} with payload: {request_data.json()}"
    )

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

@router.get("/chat_history")
def get_chat_history(
    session: dict = Depends(get_session),
    db: Session = Depends(get_db)
):
    if isinstance(session, Response):
        return session
    """
    Retrieve the chat history for the currently authenticated user and the current conversation.
    Only messages tied to the session's conversation_id will be returned.
    The returned messages have the special tokens stripped out.
    """
    username = session.get("sub")
    conversation_id = session.get("conversation_id")
    if not conversation_id:
        return []  # No conversation started yet

    records = (
        db.query(ChatHistory)
        .filter(ChatHistory.username == username, ChatHistory.conversation_id == conversation_id)
        .order_by(ChatHistory.id.asc())
        .all()
    )
    history = []
    for record in records:
        # Strip the special tokens from both the prompt and reply.
        stripped_user_prompt = strip_special_tokens(record.user_prompt) if record.user_prompt else ""
        stripped_assistant_reply = strip_special_tokens(record.assistant_reply) if record.assistant_reply else ""
        
        # Check if the record represents a system prompt by testing for the system role token.
        if record.user_prompt.startswith(SYSTEM_ROLE):
            history.append({
                "system": stripped_user_prompt,
                "timestamp": record.timestamp
            })
        else:
            history.append({
                "user": stripped_user_prompt,
                "assistant": stripped_assistant_reply,
                "timestamp": record.timestamp
            })
    return history

@router.get("/")
async def root(request: Request):
    if not request.cookies.get("access_token"):
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url="/dashboard", status_code=303)

@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, response: Response, session: dict = Depends(get_session)):
    # If get_session returns a Response object (redirect), return it
    if isinstance(session, Response):
        return session
        
    prompt_options = list(SYSTEM_PROMPTS.keys())
    return templates.TemplateResponse("index.html", {
        "request": request,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_options": prompt_options
    })

@router.post("/update_level")
def update_level(
    update: UpdateLevelRequest,
    request: Request,
    response: Response,
    session: dict = Depends(get_session)
):
    if isinstance(session, Response):
        return session
    new_level = update.new_level
    highest_level = session.get("highest_level", 1)

    # Only allow updating to levels equal to or below the highest level
    if new_level > highest_level:
        raise HTTPException(status_code=403, detail="Cannot set level higher than your current maximum level")

    # Set current_level to the selected level
    session["current_level"] = new_level

    # Reset conversation by creating a new conversation_id
    session["conversation_id"] = str(uuid.uuid4())

    # Update the JWT cookie with the new session data
    set_auth_cookie(response, session)
    return {"msg": "Level updated", "session": session}

@router.get("/get_level")
def get_level(request: Request, session: dict = Depends(get_session)):
    if isinstance(session, Response):
        return session
    return {"level": session.get("highest_level")}

@router.get("/get_current_level")
def get_current_level(session: dict = Depends(get_session)):
    if isinstance(session, Response):
        return session
    # Return both current_level and highest_level
    return {
        "current_level": session.get("current_level", session.get("highest_level", 1)),
        "highest_level": session.get("highest_level", 1)
    }

@router.get("/config")
def get_config(session: dict = Depends(get_session)):
    if isinstance(session, Response):
        return session
    if not session:
        raise HTTPException(status_code=403, detail="Invalid session")
    max_level = max(int(k.split('-')[1]) for k in SYSTEM_PROMPTS.keys())
    current_model = model.config._name_or_path
    return {
        "max_level": max_level,
        "current_model": current_model,
    }

## Auth Routes ##
oauth = OAuth()
oauth.register(
    name="github",
    client_id=os.getenv("GITHUB_CLIENT_ID"),
    client_secret=os.getenv("GITHUB_CLIENT_SECRET"),
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "user:email"},
)

@router.post("/register")
def register_user(user: RegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, password_hash=hashed_password, highest_level=1)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully."}

@router.post("/login/native")
def login_native(user: LoginRequest, response: Response, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == user.username).first()
    if not existing_user or not existing_user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(user.password, existing_user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # On login, both fields are the same by default.
    token_data = {
        "sub": existing_user.username,
        "highest_level": existing_user.highest_level,
        "current_level": existing_user.highest_level,
    }
    set_auth_cookie(response, token_data)
    return {"access_token": create_access_token(token_data)}

@router.get("/login/github")
async def login_github(request: Request):
    redirect_uri = request.url_for("auth_github")
    return await oauth.github.authorize_redirect(request, redirect_uri)

@router.get("/auth/github")
async def auth_github(request: Request, response: Response, db: Session = Depends(get_db)):
    try:
        token = await oauth.github.authorize_access_token(request)
    except OAuthError:
        raise HTTPException(status_code=400, detail="OAuth authentication failed")
    
    user_data_resp = await oauth.github.get("user", token=token)
    user_data = user_data_resp.json()
    github_id = str(user_data.get("id"))
    github_username = user_data.get("login")
    
    user = db.query(User).filter(User.oauth_provider == "github", User.oauth_id == github_id).first()
    if not user:
        user = User(
            username=github_username,
            highest_level=1,
            oauth_provider="github",
            oauth_id=github_id
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    redirect = RedirectResponse(url="/")
    
    # On login, both fields are the same by default.
    token_data = {
        "sub": user.username,
        "highest_level": user.highest_level,
        "current_level": user.highest_level
    }
    set_auth_cookie(redirect, token_data)
    return redirect

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {
        "request": request
    })

@router.post("/logout")
async def logout(response: Response):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(
        key="access_token",
        path="/",
        secure=True,
        httponly=True,
        samesite="lax"
    )
    return response

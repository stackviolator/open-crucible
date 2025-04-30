# app/routes.py

import os
import re
import uuid
import logging
from datetime import datetime

import torch
from fastapi import Request, Response, Depends, APIRouter, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from app.models import (
    model,
    tokenizer,
    LEVELS,
    load_model,
    classifier_tokenizer,
    classifier_model,
)
from app.schemas import (
    GenerationRequest,
    ChangeModelRequest,
    RegisterRequest,
    LoginRequest,
    UpdateLevelRequest,
)
from app.auth import (
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    set_auth_cookie,
    verify_password,
    get_password_hash,
)
from app.dependencies import get_session
from app.user_models import Base, User
from app.db import engine, SessionLocal

from authlib.integrations.starlette_client import OAuth, OAuthError
from passlib.context import CryptContext

# Create the router and template objects
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

DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"


def strip_special_tokens(text: str) -> str:
    # Build the list from the defined constants instead of hard coding.
    tokens = [
        SYSTEM_ROLE,
        SYSTEM_ROLE_END,
        USER_ROLE,
        USER_ROLE_END,
        ASSISTANT_ROLE,
        ASSISTANT_ROLE_END,
    ]
    for token in tokens:
        text = text.replace(token, "")
    return text.strip()


# --- New ChatHistory Model ---
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    conversation_id = Column(String, index=True)  # Ties records to a conversation UUID.
    user_prompt = Column(
        String
    )  # Stores the user prompt or system prompt (with tokens).
    assistant_reply = Column(String)  # Stores the assistant reply (with tokens).
    timestamp = Column(String)  # Stored as ISO string.


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


# -----------------------------
# Password Validation Function
# -----------------------------
def validate_password(password: str):
    """
    Validates the password against a set of rules:
      - Minimum length of 8 characters.
      - Contains at least one uppercase letter.
      - Contains at least one lowercase letter.
      - Contains at least one digit.
      - Contains at least one special character.
      - Does not contain any whitespace.
      - Less than 128 characters.
    """
    if len(password) < 8:
        raise HTTPException(
            status_code=400, detail="Password must be at least 8 characters long."
        )
    if not re.search(r"[A-Z]", password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least one uppercase letter.",
        )
    if not re.search(r"[a-z]", password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least one lowercase letter.",
        )
    if not re.search(r"\d", password):
        raise HTTPException(
            status_code=400, detail="Password must contain at least one digit."
        )
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least one special character.",
        )
    if re.search(r"\s", password):
        raise HTTPException(
            status_code=400, detail="Password cannot contain any whitespace."
        )
    if len(password) > 128:
        raise HTTPException(
            status_code=400, detail="Password cannot be longer than 128 characters."
        )


# --- Generate Text Endpoint ---
@router.post("/generate")
def generate_text(
    request_data: GenerationRequest,
    request: Request,
    response: Response,
    session: dict = Depends(get_session),
    db: Session = Depends(get_db),
):
    if isinstance(session, Response):
        return session
    """
    Generate text using the language model based on the user's prompt.
    Records the conversation (system prompt, user prompt, and assistant reply) in the database,
    tied to a conversation UUID.
    """
    # Convert the prompt choice (e.g. "level-1") to a numeric level.
    try:
        level_number = int(request_data.system_prompt_choice.split("-")[-1])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid prompt level format")

    if level_number > session.get("highest_level", 0):
        raise HTTPException(
            status_code=403, detail="You do not have access to this prompt."
        )

    # Look up the selected level from LEVELS using the level number.
    selected_level = next(
        (lvl for lvl in LEVELS.values() if lvl.get("index") == level_number), None
    )
    selected_system_prompt = selected_level.get("system_prompt")

    if DEBUG_MODE:
        print(f"DEBUG: Selected level index: {level_number}")
        print(f"DEBUG: Selected system prompt: {selected_system_prompt}")

    # --- Conversation UUID Management ---
    conversation_reset = session.get("reset_conversation", False)
    conversation_id = session.get("conversation_id")
    timestamp = datetime.utcnow().isoformat()
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
            timestamp=timestamp,
        )
        db.add(system_record)
        db.commit()
        if DEBUG_MODE:
            print(
                f"DEBUG: New conversation started. conversation_id set to: {conversation_id}"
            )

    # --- Load Existing Conversation History from DB ---
    conversation_records = (
        db.query(ChatHistory)
        .filter(
            ChatHistory.username == session["sub"],
            ChatHistory.conversation_id == conversation_id,
        )
        .order_by(ChatHistory.id.asc())
        .all()
    )
    if DEBUG_MODE:
        print(
            f"DEBUG: Loaded {len(conversation_records)} records for conversation_id {conversation_id}"
        )

    # Build final prompt starting with BOS token and system prompt
    final_prompt = (
        f"{BOS_TOKEN}\n{SYSTEM_ROLE}{selected_system_prompt}{SYSTEM_ROLE_END}\n"
    )

    # Add each record from conversation history
    for record in conversation_records:
        if record.user_prompt:
            final_prompt += record.user_prompt + "\n"
        if record.assistant_reply:
            final_prompt += record.assistant_reply + "\n"

    # Add the new user prompt and assistant tag
    final_prompt += f"<user>{request_data.user_prompt}</user>\n<assistant>"

    if DEBUG_MODE:
        print("DEBUG: Final prompt sent to model:\n", final_prompt)

    # --- Generate Model Output ---
    prompt_inputs = tokenizer(final_prompt, return_tensors="pt", padding=True)
    input_ids = prompt_inputs["input_ids"].to(model.device)
    attention_mask = prompt_inputs["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    # For token visualization
    user_input_tokens = [tokenizer.decode(token) for token in input_ids[0]]

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=(
                request_data.max_new_tokens
                if request_data.max_new_tokens <= 200
                else 200
            ),
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.75,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.1,
        )

    # Process generated tokens
    output_token_ids = generated_ids[0][prompt_len:]
    generated_text_only = tokenizer.decode(output_token_ids, skip_special_tokens=True)
    if "</assistant>" in generated_text_only:
        generated_text_only = generated_text_only.split("</assistant>")[0]

    # For visualization: re-encode the generated text.
    output_tokens_ids = tokenizer.encode(generated_text_only)
    output_tokens = [tokenizer.decode(token) for token in output_tokens_ids]

    if DEBUG_MODE:
        print("DEBUG: Generated text only:", generated_text_only)

    # --- Jailbreak Check and Level Update ---
    lower_output_text = generated_text_only.lower()
    lower_input_text = request_data.user_prompt.lower()
    flag_pattern = selected_level.get(
        "flag_pattern", None
    )  # Retrieve the "flag" value from the level content
    output_blacklist_pattern = selected_level.get("guardrails", {}).get(
        "output_blacklist_pattern", None
    )
    input_blacklist_pattern = selected_level.get("guardrails", {}).get(
        "input_blacklist_pattern", None
    )

    jailbreak_success = False
    blacklist_violation = False

    if flag_pattern:
        try:
            jailbreak_success = (
                re.search(flag_pattern, lower_output_text, flags=re.IGNORECASE)
                is not None
            )
        except re.error as regex_error:
            logging.warning(
                f"Invalid regex pattern for flag: {flag_pattern}. Error: {regex_error}"
            )

    if output_blacklist_pattern:
        try:
            blacklist_violation = (
                re.search(
                    output_blacklist_pattern, lower_output_text, flags=re.IGNORECASE
                )
                is not None
            )
        except re.error as regex_error:
            logging.warning(
                f"Invalid regex pattern for blacklist: {output_blacklist_pattern}. Error: {regex_error}"
            )

    if input_blacklist_pattern:
        try:
            blacklist_violation = (
                re.search(
                    input_blacklist_pattern, lower_input_text, flags=re.IGNORECASE
                )
                is not None
            )
        except re.error as regex_error:
            logging.warning(
                f"Invalid regex pattern for blacklist: {input_blacklist_pattern}. Error: {regex_error}"
            )

    # If a blacklist violation is detected, log it and return an error response
    if blacklist_violation:
        logging.warning(f"Blacklist violation detected: {generated_text_only}")
        return {
            "system_prompt": session.get("highest_level"),
            "combined_prompt": final_prompt,
            "generated_text_only": generated_text_only,
            "jailbreak_success": False,
            "jailbreak_detected": True,
            "blacklist_violation": blacklist_violation,
            "user_tokens": user_input_tokens,  # Token information.
            "output_tokens": output_tokens,  # Token information.
        }

    try:
        if jailbreak_success and level_number >= session.get("highest_level", 0):
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
                timestamp=timestamp,
            )
            db.add(system_record)
            db.commit()
            if DEBUG_MODE:
                print(
                    "DEBUG: Conversation reset after level up. New conversation_id:",
                    conversation_id,
                )
    except (AttributeError, IndexError, ValueError) as e:
        logging.warning(
            f"Error processing level update: {str(e)}. system_prompt_choice: {request_data.system_prompt_choice}, session: {session}"
        )

    # Store the conversation turn.
    new_record = ChatHistory(
        username=session["sub"],
        conversation_id=conversation_id,
        user_prompt=f"<user>{request_data.user_prompt}</user>",
        assistant_reply=f"<assistant>{generated_text_only}</assistant>",
        timestamp=timestamp,
    )
    db.add(new_record)
    db.commit()

    if DEBUG_MODE:
        print(
            "DEBUG: New conversation turn stored for conversation_id:", conversation_id
        )

    return {
        "system_prompt": session.get("highest_level"),
        "combined_prompt": final_prompt,
        "generated_text_only": generated_text_only,
        "jailbreak_success": jailbreak_success,
        "jailbreak_detected": False,
        "blacklist_violation": None,
        "user_tokens": user_input_tokens,  # Token information.
        "output_tokens": output_tokens,  # Token information.
    }


@router.get("/get_prompt")
def get_prompt(key: int, session: dict = Depends(get_session)):
    """
    Retrieve the system prompt corresponding to the specified level (by its index).
    """
    if key > session.get("highest_level", 0):
        raise HTTPException(
            status_code=403, detail="You do not have access to this prompt."
        )
    selected_level = next(
        (lvl for lvl in LEVELS.values() if lvl.get("index") == key), None
    )
    prompt_text = (
        selected_level.get("system_prompt", "Prompt not found.")
        if selected_level
        else "Prompt not found."
    )
    return {"prompt_text": prompt_text}


@router.post("/change_model")
def change_model(
    request_data: ChangeModelRequest,
    request: Request,
    session: dict = Depends(get_session),
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
    session: dict = Depends(get_session), db: Session = Depends(get_db)
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
        return []  # No conversation started yet.

    records = (
        db.query(ChatHistory)
        .filter(
            ChatHistory.username == username,
            ChatHistory.conversation_id == conversation_id,
        )
        .order_by(ChatHistory.id.asc())
        .all()
    )
    history = []
    for record in records:
        stripped_user_prompt = (
            strip_special_tokens(record.user_prompt) if record.user_prompt else ""
        )
        stripped_assistant_reply = (
            strip_special_tokens(record.assistant_reply)
            if record.assistant_reply
            else ""
        )

        if record.user_prompt.startswith(SYSTEM_ROLE):
            history.append(
                {"system": stripped_user_prompt, "timestamp": record.timestamp}
            )
        else:
            history.append(
                {
                    "user": stripped_user_prompt,
                    "assistant": stripped_assistant_reply,
                    "timestamp": record.timestamp,
                }
            )
    return history


@router.get("/")
async def root(request: Request):
    if not request.cookies.get("access_token"):
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url="/dashboard", status_code=303)


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request, response: Response, session: dict = Depends(get_session)
):
    if isinstance(session, Response):
        return session
    # Build prompt options from LEVELS sorted by index.
    prompt_options = [
        lvl["name"] for lvl in sorted(LEVELS.values(), key=lambda l: l.get("index", 0))
    ]

    # Get the current level's system prompt
    current_level = session.get("current_level", session.get("highest_level", 1))
    selected_level = next(
        (lvl for lvl in LEVELS.values() if lvl.get("index") == current_level), None
    )
    system_prompt = selected_level.get("system_prompt", "") if selected_level else ""

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "system_prompt": system_prompt,
            "prompt_options": prompt_options,
        },
    )


@router.post("/update_level")
def update_level(
    update: UpdateLevelRequest,
    request: Request,
    response: Response,
    session: dict = Depends(get_session),
):
    if isinstance(session, Response):
        return session
    new_level = update.new_level
    highest_level = session.get("highest_level", 1)

    if new_level > highest_level:
        raise HTTPException(
            status_code=403,
            detail="Cannot set level higher than your current maximum level",
        )

    session["current_level"] = new_level
    session["conversation_id"] = str(uuid.uuid4())
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
    return {
        "current_level": session.get("current_level", session.get("highest_level", 1)),
        "highest_level": session.get("highest_level", 1),
    }


@router.get("/config")
def get_config(session: dict = Depends(get_session)):
    if isinstance(session, Response):
        return session
    if not session:
        raise HTTPException(status_code=403, detail="Invalid session")
    max_level = max(lvl.get("index", 0) for lvl in LEVELS.values())
    current_model = model.config._name_or_path
    return {
        "max_level": max_level,
        "current_model": current_model,
    }


@router.get("/levels")
def get_levels(session: dict = Depends(get_session)):
    if isinstance(session, Response):
        return session

    highest_level = session.get("highest_level", 1)
    # Get all levels, modifying locked ones
    all_levels = [
        {
            "index": level["index"],
            "name": (
                level["name"]
                if level["index"] <= highest_level
                else f"🔒 Level {level['index']} (Locked)"
            ),
            "description": (
                level["description"]
                if level["index"] <= highest_level
                else "Complete the previous level to unlock this challenge!"
            ),
            "difficulty": (
                "Completed"
                if level["index"] < highest_level
                else (
                    level.get("difficulty", "Unknown")
                    if level["index"] <= highest_level
                    else "Unknown"
                )
            ),
            "locked": level["index"] > highest_level,
        }
        for level in sorted(LEVELS.values(), key=lambda l: l.get("index", 0))
    ]

    total_levels = max(level.get("index", 0) for level in LEVELS.values())

    return {"levels": all_levels, "total_levels": total_levels}


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
    validate_password(user.password)

    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(
        username=user.username, password_hash=hashed_password, highest_level=1
    )
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
async def auth_github(
    request: Request, response: Response, db: Session = Depends(get_db)
):
    try:
        token = await oauth.github.authorize_access_token(request)
    except OAuthError:
        raise HTTPException(status_code=400, detail="OAuth authentication failed")

    user_data_resp = await oauth.github.get("user", token=token)
    user_data = user_data_resp.json()
    github_id = str(user_data.get("id"))
    github_username = user_data.get("login")

    user = (
        db.query(User)
        .filter(User.oauth_provider == "github", User.oauth_id == github_id)
        .first()
    )
    if not user:
        user = User(
            username=github_username,
            highest_level=1,
            oauth_provider="github",
            oauth_id=github_id,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    redirect = RedirectResponse(url="/")
    token_data = {
        "sub": user.username,
        "highest_level": user.highest_level,
        "current_level": user.highest_level,
    }
    set_auth_cookie(redirect, token_data)
    return redirect


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.post("/logout")
async def logout(response: Response):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(
        key="access_token", path="/", secure=True, httponly=True, samesite="lax"
    )
    return response

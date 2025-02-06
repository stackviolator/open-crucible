# Open Crucible

![Open Crucible](img/home-page.png)

Open Crucible is a playground for experimenting with prompt injections and understanding how different prompt components interact with language models. Built with [FastAPI](https://fastapi.tiangolo.com/) and powered by [Hugging Face Transformers](https://huggingface.co/transformers/), this project provides a web interface to test and visualize how system prompts, user prompts, and tokenization influence model outputs.

---

## Features

- **Customizable Prompts:** Combines a hard-coded system prompt with user-provided input in a structured format.
- **Token Visualization:** Displays tokenized representations of both user input and model output.
- **Web Interface:** Interactive HTML frontend to experiment with prompt injections.

---

## Project Structure

- **`models.py`**  
  Loads the tokenizer and model, sets up special tokens (including a custom pad token), and configures the model.

- **`routes.py`**  
  Defines API endpoints using FastAPI. The `/generate` endpoint constructs a combined prompt (system + user) and generates text, while the root endpoint serves the HTML interface.

- **`main.py`**  
  The FastAPI application entrypoint. It includes the routes and runs the server using Uvicorn.

- **`schemas.py`**  
  Contains Pydantic models for request validation, such as the `GenerationRequest`.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [PyTorch](https://pytorch.org/) (with CUDA support if you want GPU acceleration)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [Transformers](https://huggingface.co/transformers/)
- Other dependencies (see `requirements.txt` if provided)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/stackviolator/open-crucible.git
   cd open-crucible
   ```

2. **Create a venv**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Start the server with:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --proxy-headers
```

Once running, open http://localhost:8000 in your browser.
import os
import torch
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

app = FastAPI(title="Text2 Chat AI Qwen API")

# Configure CORS so that the frontend can call the API directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
print(f"Loading model & tokenizer: {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)
if device == "cpu":
    model.to("cpu")

print("Model and tokenizer loaded successfully!")

# Retrieve API key from environment variable (set this in HF Space Settings)
API_KEY = os.environ.get("API_KEY", "")
if API_KEY:
    print("API Key protection is ENABLED.")
else:
    print("API Key protection is DISABLED (anyone can access this Space).")

@app.get("/")
def home():
    return {
        "status": "online",
        "model": model_name,
        "device": device,
        "api_key_required": bool(API_KEY)
    }

@app.post("/chat")
async def chat(request: Request):
    # API key check if configured
    if API_KEY:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startsWith("Bearer ") or auth_header.split(" ")[1] != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API Key")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    messages = data.get("messages", [])
    prompt = data.get("prompt", "")
    system_prompt = data.get("system_prompt", "Always reply in English only, no matter what language the user writes in. Answer briefly under 700 characters in a teen style, and play as you are text2 chat AI of text2.")

    # Convert single prompt to chat history structure if 'messages' is empty
    if not messages:
        if not prompt:
            raise HTTPException(status_code=400, detail="Either 'messages' or 'prompt' is required")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

    # Apply the Qwen chat template to structure instructions properly
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize input
    inputs = tokenizer([text], return_tensors="pt").to(device)

    # Setup transformers streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generation arguments
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )

    # Run model generation in a background thread to prevent blocking FastAPI
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    async def event_generator():
        try:
            for new_text in streamer:
                if new_text:
                    yield f"data: {json.dumps({'text': new_text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

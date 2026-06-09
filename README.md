---
title: Text2 Chat AI Qwen API
emoji: 💬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Qwen2.5-0.5B-Instruct API Space

This Hugging Face Space runs a FastAPI backend with streaming response for Qwen2.5-0.5B-Instruct.

## API Usage

- **Endpoint**: `POST /chat`
- **Headers**:
  - `Content-Type: application/json`
  - `Authorization: Bearer <API_KEY>` (Only required if you set the `API_KEY` secret variable in the Space Settings)
- **Request Body**:
  ```json
  {
    "prompt": "Hello",
    "system_prompt": "Optional system prompt"
  }
  ```
  or OpenAI format:
  ```json
  {
    "messages": [
      {"role": "system", "content": "Always reply in English only..."},
      {"role": "user", "content": "Hello"}
    ]
  }
  ```

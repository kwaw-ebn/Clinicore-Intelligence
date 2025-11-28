# Chatbot integration (high-level)
- You can integrate a chatbot UI in the dashboard that calls a backend /chat endpoint.
- Backend can proxy to an LLM service (OpenAI, local LLM, or HuggingFace Inference API).
- Example frontend flow:
  1. User types query
  2. Frontend posts to /chat with user message and session ID
  3. Backend authenticates and calls LLM, returns response
- Make sure to implement rate-limiting and usage logging.

"""Main module for Fragaria - Chain of Thought Reasoning API"""

import os
import asyncio
import random
import json
import time
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import yaml

from .core import FragariaCore

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

try:
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    # Initialize Fragaria core
    fragaria_core = FragariaCore(config_path)
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please update your config.yaml file with valid API keys.")
    fragaria_core = None
except FileNotFoundError:
    print(f"Configuration file not found at {config_path}")
    print("Please create a config.yaml file with your API keys.")
    fragaria_core = None
except Exception as e:
    print(f"Error initializing Fragaria core: {e}")
    fragaria_core = None

# FastAPI setup
app = FastAPI(
    title="Chain of Thought Reasoning API",
    description="An API for performing chain of thought reasoning using various LLM providers.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["chat"])
async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    """
    Perform a chat completion using the Chain of Thought reasoning process.
    """
    if fragaria_core is None:
        raise HTTPException(status_code=500, detail="Fragaria core not initialized. Please check your configuration.")
    
    if request.model not in ["faragia-dev"]:
        raise HTTPException(status_code=400, detail="Unsupported model")
    
    system_prompt = next((msg['content'] for msg in request.messages if msg['role'] == 'system'), "")
    
    # Extract user messages and assistant replies in order
    conversation = []
    for msg in request.messages:
        if msg['role'] in ['user', 'assistant']:
            conversation.append(f"{msg['role'].capitalize()}: {msg['content']}")
    
    # Join the conversation history
    conversation_history = "\n".join(conversation)
    
    # Append the conversation history to the system prompt
    full_system_prompt = f"{system_prompt}\n\nConversation history:\n{conversation_history}"
    
    # Get the latest user message
    latest_user_message = next((msg['content'] for msg in reversed(request.messages) if msg['role'] == 'user'), "")
    
    result = await fragaria_core.parallel_cot_reasoning(latest_user_message, full_system_prompt)
    
    response = ChatCompletionResponse(
        id=f"chatcmpl-{random.randint(1000000, 9999999)}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(result, indent=2)
                },
                "finish_reason": "stop"
            }
        ],
        usage={
            "prompt_tokens": len(conversation_history.split()) + len(latest_user_message.split()),
            "completion_tokens": len(json.dumps(result).split()),
            "total_tokens": len(conversation_history.split()) + len(latest_user_message.split()) + len(json.dumps(result).split())
        }
    )
    
    background_tasks.add_task(log_interaction, latest_user_message, result, conversation_history)
    
    return response

async def log_interaction(text: str, result: Dict[str, any], conversation_history: str):
    # Implement logging logic here (e.g., to a database or file)
    # Now includes the conversation_history in the log
    pass

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Chain of Thought Reasoning API",
        version="1.0.0",
        description="An API for performing chain of thought reasoning using various LLM providers.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Mount static files for frontend
frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "public")
build_path = os.path.join(frontend_path, "build")
if os.path.exists(build_path):
    app.mount("/build", StaticFiles(directory=build_path), name="static")

@app.get("/")
async def read_index():
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not available"}

def run_server(host: str = None, port: int = None):
    """Run the Fragaria API server"""
    import uvicorn
    
    # Use config values if not provided
    if host is None:
        host = config["server"]["host"]
    if port is None:
        port = config["server"]["port"]
        
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()
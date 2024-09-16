import os
import asyncio
import aiohttp
import openai
from typing import List, Dict, Optional, Any
import json
from collections import defaultdict
import random
import sqlite3
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import time
import yaml
import math
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

api_key = None

# Set up the OpenAI client based on the selected provider
LLM_PROVIDER = config["llm_provider"]
if LLM_PROVIDER == "openai":
    api_key = config["openai_api_key"]
    client = openai.OpenAI(api_key=config["openai_api_key"])
elif LLM_PROVIDER == "groq":
    api_key = config["groq_api_key"]
    client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=config["groq_api_key"]
    )
elif LLM_PROVIDER == "together":
    api_key = config["together_api_key"]
    client = openai.OpenAI(
        api_key=config["together_api_key"],
        base_url="https://api.together.xyz/v1",
    )
else:
    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

# Model configuration
model_config = config["model_config"][LLM_PROVIDER]

# Database setup
def init_db():
    conn = sqlite3.connect(config["database"]["path"])
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cot_paths
                 (problem_type TEXT, method TEXT, score REAL, uses INTEGER)''')
    conn.commit()
    conn.close()

init_db()

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

async def call_openai_api(model: str, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content.strip()

async def classify_or_create_problem_type(text: str) -> str:
    conn = sqlite3.connect('cot_database.db')
    c = conn.cursor()
    c.execute("SELECT DISTINCT problem_type FROM cot_paths")
    known_problem_types = set(row[0] for row in c.fetchall())
    conn.close()

    system_prompt = "You are an AI assistant specialized in classifying problem types. Your task is to either classify the given text into an existing problem type or create a new suitable problem type if none of the existing ones fit well. You will respond in JSON."

    example_json = json.dumps({"problem_type": "<problem type>"})
    
    if not known_problem_types:
        user_prompt = f"Create a suitable problem type classification for the following text. Respond with only the problem type name as a JSON. Example json is {example_json} \n\nText: {text}"
    else:
        types_str = ", ".join(known_problem_types)
        user_prompt = f"Classify the following text into one of these problem types: {types_str}. If none fit well, create a new suitable problem type. Respond with only the problem type name as a JSON. Example json is {example_json} \n\nText: {text}"
    
    problem_type = await call_openai_api(model_config["classify"], system_prompt, user_prompt)
    problem_type = json.loads(problem_type)["problem_type"]
    return problem_type

async def generate_cot_paths(text: str, problem_type: str) -> List[Dict[str, any]]:
    example_json = json.dumps({"approaches": [{"method":"<method name>","description":"<method description>","steps":["<detailed step 1>","<detailed step 2>","<detailed step 3>"]}]})
    system_prompt = "You are an AI assistant specialized in generating diverse chain of thought approaches for problem-solving. You will respond in JSON."
    user_prompt = f"Generate a list of 3 different chain of thought approaches to analyze the following {problem_type} problem. Respond in JSON. Example {example_json} \n\nProblem: {text}"
    
    response = await call_openai_api(model_config["generate"], system_prompt, user_prompt)
    response = json.loads(response)
    return response["approaches"]

async def run_cot_path(session: aiohttp.ClientSession, text: str, path: Dict[str, any], problem_type: str, system_prompt: str) -> Dict:
    full_system_prompt = f"{system_prompt}\nYou are an AI assistant specialized in analyzing {problem_type} problems using specific chain of thought approaches. Your task is to apply the given approach to analyze the problem."
    user_prompt = f"Analyze the following problem using this chain of thought approach: {json.dumps(path)}\n\nProblem: {text}"
    
    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_config["analyze"],
            "messages": [
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
    ) as response:
        result = await response.json()
        return {"method": path, "result": result['choices'][0]['message']['content']}

async def combine_results(results: List[Dict], problem_type: str, system_prompt: str) -> str:
    example_json = {"results": {"<method name>": "<text paragraph>"}}
    full_system_prompt = f"{system_prompt}\nYou are an AI assistant specialized in combining and summarizing multiple analysis results for {problem_type} problems. Your task is to synthesize the given results into a coherent summary. You will respond in JSON."
    results_text = "\n\n".join([f"Method: {json.dumps(r['method'])}\nResult: {r['result']}" for r in results])
    user_prompt = f"Combine and summarize the following analysis results as a JSON with a results key mapped to a text paragraph per result. Do not modify the method name. Example {json.dumps(example_json)} \n\n{results_text}"
    
    final_result = await call_openai_api(model_config["combine"], full_system_prompt, user_prompt)
    final_result = json.loads(final_result)
    return final_result

async def evaluate_result(text: str, result: str, problem_type: str, system_prompt: str) -> Dict[str, float]:
    example_json = {"<method 1>": "<score>", "<method 2>": "<score>", "<method 3>": "<score>"}
    full_system_prompt = f"{system_prompt}\nYou are an AI assistant specialized in evaluating the quality of analysis for {problem_type} problems. Your task is to rate the given result on a scale from 0 to 10, where 10 is the highest quality."
    user_prompt = f"On a scale of 0 to 10, how well does this result answer or analyze the given problem? Provide only the numerical score in the JSON format for every type of result. Please retain the method name. Example {json.dumps(example_json)} \n\nProblem: {text}\n\nResult: {json.dumps(result)}"
    
    scores = await call_openai_api(model_config["evaluate"], full_system_prompt, user_prompt)
    scores = json.loads(scores)
    return scores

async def update_cot_scores(problem_type: str, paths: List[Dict[str, any]], scores: Dict[str, float]):
    conn = sqlite3.connect('cot_database.db')
    c = conn.cursor()
    for path in paths:
        method = path['method']
        c.execute("SELECT score, uses FROM cot_paths WHERE problem_type = ? AND method = ?", (problem_type, method))
        result = c.fetchone()
        if result:
            current_score, current_uses = result
            new_score = (current_score * current_uses + float(scores[method])) / (current_uses + 1)
            new_uses = current_uses + 1
            c.execute("UPDATE cot_paths SET score = ?, uses = ? WHERE problem_type = ? AND method = ?",
                      (new_score, new_uses, problem_type, method))
        else:
            # For new paths, insert with the initial score
            c.execute("INSERT INTO cot_paths (problem_type, method, score, uses) VALUES (?, ?, ?, ?)",
                      (problem_type, method, float(scores[method]), 1))
    conn.commit()
    conn.close()

def select_cot_paths(problem_type: str, n: int = 3) -> List[Dict[str, any]]:
    conn = sqlite3.connect('cot_database.db')
    c = conn.cursor()
    c.execute("SELECT method, score, uses FROM cot_paths WHERE problem_type = ?", (problem_type,))
    type_scores = {row[0]: {"score": row[1], "uses": row[2]} for row in c.fetchall()}
    conn.close()
    
    total_uses = sum(data["uses"] for data in type_scores.values())
    
    # Calculate UCB scores
    ucb_scores = {}
    for method, data in type_scores.items():
        if data["uses"] == 0:
            ucb_scores[method] = float('inf')  # Ensure new methods are tried
        else:
            average_score = data["score"] / data["uses"]
            exploration_term = math.sqrt(2 * math.log(total_uses) / data["uses"])
            ucb_scores[method] = average_score + exploration_term
    
    # Select top n paths based on UCB scores
    sorted_paths = sorted(ucb_scores.items(), key=lambda x: x[1], reverse=True)
    top_paths = [{"method": method, "score": type_scores[method]["score"], "uses": type_scores[method]["uses"]} 
                 for method, _ in sorted_paths[:n]]
    
    # If we don't have enough paths, add new exploratory paths
    while len(top_paths) < n:
        top_paths.append({"method": f"New exploratory path {random.randint(1, 1000)}", "score": 0, "uses": 0})
    
    return top_paths

async def adapt_cot_path(path: Dict[str, any], problem_type: str, text: str, system_prompt: str) -> Dict[str, any]:
    full_system_prompt = f"{system_prompt}\nYou are an AI assistant specialized in adapting chain of thought approaches for new problems. Your task is to modify the given approach to better suit the current problem while maintaining its core strategy."
    user_prompt = f"Adapt the following chain of thought approach for the current {problem_type} problem:\n\nOriginal approach: {json.dumps(path)}\n\nCurrent problem: {text}\n\nProvide the adapted approach in the same JSON format as the original."
    
    adapted_path = await call_openai_api(model_config["adapt"], full_system_prompt, user_prompt)
    adapted_path = json.loads(adapted_path)
    return adapted_path

async def parallel_cot_reasoning(text: str, system_prompt: str) -> Dict[str, any]:
    problem_type = await classify_or_create_problem_type(text)
    
    stored_paths = select_cot_paths(problem_type)
    cot_paths = []
    for path in stored_paths:
        if path["uses"] == 0:
            # For new paths, generate a new approach
            new_path = await generate_cot_paths(text, problem_type)
            cot_paths.extend(new_path)
        else:
            # For existing paths, adapt them
            adapted_path = await adapt_cot_path(path, problem_type, text, system_prompt)
            cot_paths.append(adapted_path)
    
    async with aiohttp.ClientSession() as session:
        tasks = [run_cot_path(session, text, path, problem_type, system_prompt) for path in cot_paths]
        results = await asyncio.gather(*tasks)
        
    final_result = await combine_results(results, problem_type, system_prompt)
    scores = await evaluate_result(text, final_result, problem_type, system_prompt)
    highest_score_method = max(scores, key=scores.get)

    await update_cot_scores(problem_type, cot_paths, scores)
    
    #return {
    #    "problem_type": problem_type,
    #    "results": final_result,
    #    "scores": scores,
    #    "cot_paths": cot_paths
    #}

    return final_result["results"][highest_score_method]

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["chat"])
async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    """
    Perform a chat completion using the Chain of Thought reasoning process.
    """
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
    
    result = await parallel_cot_reasoning(latest_user_message, full_system_prompt)
    
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

app.mount("/build", StaticFiles(directory="frontend/public/build"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('frontend/public/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"])
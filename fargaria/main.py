import asyncio
import aiohttp
import openai
from typing import List, Dict, Tuple
import json
from collections import defaultdict
import random

# Replace with your actual OpenAI API key
api_key = "UseOwnAPIKey"

# Store CoT path scores for each problem type
cot_scores = defaultdict(lambda: defaultdict(lambda: {"score": 0, "uses": 0}))

# Store known problem types
known_problem_types = set()

# Model configuration
model_config = {
    "classify": "gpt-4o",
    "generate": "gpt-4o",
    "analyze": "gpt-4o",
    "combine": "gpt-4o",
    "evaluate": "gpt-4o"
}

async def call_openai_api(model: str, system_prompt: str, user_prompt: str) -> str:
    client = openai.OpenAI(api_key=api_key)
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
    system_prompt = "You are an AI assistant specialized in classifying problem types. Your task is to either classify the given text into an existing problem type or create a new suitable problem type if none of the existing ones fit well. You will respond in JSON."

    example_json = json.dumps({"problem_type": "<problem type>"})
    
    if not known_problem_types:
        user_prompt = f"Create a suitable problem type classification for the following text. Respond with only the problem type name as a JSON. Example json is {example_json} \n\nText: {text}"
    else:
        types_str = ", ".join(known_problem_types)
        user_prompt = f"Classify the following text into one of these problem types: {types_str}. If none fit well, create a new suitable problem type. Respond with only the problem type name as a JSON.Example json is {example_json} \n\nText: {text}"
    
    problem_type = await call_openai_api(model_config["classify"], system_prompt, user_prompt)
    problem_type = json.loads(problem_type)  # Ensure it's valid JSON
    known_problem_types.add(problem_type["problem_type"])
    return problem_type["problem_type"]

async def generate_cot_paths(text: str, problem_type: str) -> List[str]:
    example_json = json.dumps({"approaches": [{"method":"<method name>","description":"<method description>","steps":["<detailed step 1>","<detailed step 2>","<detailed step 3>"]}]})
    system_prompt = "You are an AI assistant specialized in generating diverse chain of thought approaches for problem-solving. You will respond in JSON."
    user_prompt = f"Generate list of 3 different chain of thought approaches to analyze the following {problem_type} problem. Respond in JSON. Example {example_json} \n\nProblem: {text}"
    
    response = await call_openai_api(model_config["generate"], system_prompt, user_prompt)
    response = json.loads(response)
    return response["approaches"]

async def run_cot_path(session: aiohttp.ClientSession, text: str, path: str, problem_type: str) -> Dict:
    system_prompt = f"You are an AI assistant specialized in analyzing {problem_type} problems using specific chain of thought approaches. Your task is to apply the given approach to analyze the problem."
    user_prompt = f"Analyze the following problem using this chain of thought approach: {path}\n\nProblem: {text}"
    
    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_config["analyze"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
    ) as response:
        result = await response.json()
        return {"path": path, "result": result['choices'][0]['message']['content']}

async def combine_results(results: List[Dict], problem_type: str) -> str:
    example_json = {"results": {"<method name>": "<text paragraph>"}}
    system_prompt = f"You are an AI assistant specialized in combining and summarizing multiple analysis results for {problem_type} problems. Your task is to synthesize the given results into a coherent summary. You will respond in JSON"
    results_text = "\n\n".join([f"Path: {r['path']}\nResult: {r['result']}" for r in results])
    user_prompt = f"Combine and summarize the following analysis results as a json with a results key mapped to a text paragraph per result. Example {example_json} \n\n{results_text}"
    
    final_result = await call_openai_api(model_config["combine"], system_prompt, user_prompt)
    final_result = json.loads(final_result)
    return final_result

async def evaluate_result(text: str, result: str, problem_type: str) -> float:
    example_json = {"<method 1>": "<score>", "<method 2>": "<score>", "<method 3>": "<score>"}
    system_prompt = f"You are an AI assistant specialized in evaluating the quality of analysis for {problem_type} problems. Your task is to rate the given result on a scale from 0 to 10, where 10 is the highest quality."
    user_prompt = f"On a scale of 0 to 10, how well does this result answer or analyze the given problem? Provide only the numerical score in the JSON format for every type of result.Please retain the method name. Example {example_json} \n\nProblem: {text}\n\nResult: {result}"
    
    scores = await call_openai_api(model_config["evaluate"], system_prompt, user_prompt)
    scores = json.loads(scores)
    return scores

async def update_cot_scores(problem_type: str, paths: List[str], scores: dict):
    for path in paths:
        cot_scores[problem_type][path['method']]["score"] += float(scores[path['method']])
        cot_scores[problem_type][path['method']]["uses"] += 1

def select_cot_paths(problem_type: str, n: int = 3, exploration_rate: float = 0.2) -> List[str]:
    type_scores = cot_scores[problem_type]
    
    # Exploitation: Select top performing paths
    sorted_paths = sorted(type_scores.items(), key=lambda x: x[1]["score"] / max(x[1]["uses"], 1), reverse=True)
    top_paths = [path for path, _ in sorted_paths[:n]]
    
    # Exploration: Randomly replace some paths with new ones
    for i in range(n):
        if random.random() < exploration_rate:
            top_paths[i] = f"New exploratory path {random.randint(1, 1000)}"
    
    return top_paths

async def parallel_cot_reasoning(text: str) -> Tuple[str, float, str]:
    problem_type = await classify_or_create_problem_type(text)
    
    if not cot_scores[problem_type]:  # If no scores for this problem type, generate new paths
        cot_paths = await generate_cot_paths(text, problem_type)
    else:
        cot_paths = select_cot_paths(problem_type)
    
    async with aiohttp.ClientSession() as session:
        tasks = [run_cot_path(session, text, json.dumps(path), problem_type) for path in cot_paths]
        results = await asyncio.gather(*tasks)
        
    final_result = await combine_results(results, problem_type)
    scores = await evaluate_result(text, final_result, problem_type)
    highest_score_method = max(scores, key=scores.get)
    
    await update_cot_scores(problem_type, cot_paths, scores)
    result = final_result["results"][highest_score_method]

    return final_result, scores, problem_type, result

async def main():
    texts = [
        #"What are the economic implications of rising inflation rates?",
        #"Design a sustainable urban transportation system for a city of 1 million people.",
        #"Solve the equation: 3x^2 + 7x - 2 = 0",
        "How many 'r's in strawberry?",
        """A princess is as old as the prince will be when the princess is twice as old as the prince was when the princess’s age was half the sum of their present age. What is the age of prince and princess? Provide all solutions to that question.""",
        #"If all A are B, and some B are C, what can we conclude about A and C?",
        #"How might climate change affect global food security in the next 50 years?",
        #"Compose a haiku about artificial intelligence.",
        #"What are the ethical considerations of genetic engineering in humans?",
    ]
    
    for text in texts:
        result, score, problem_type, output = await parallel_cot_reasoning(text)
        print(f"Problem: {text}")
        print(f"Problem Type: {problem_type}")
        print(f"Result: {result}")
        print(f"Score: {score}")
        print(f"Output: {output}")
        print("Current CoT Scores for this problem type:")
        print(json.dumps(cot_scores[problem_type], indent=2))
        print("\n" + "="*50 + "\n")
    
    print("Known Problem Types:")
    print(json.dumps(list(known_problem_types), indent=2))
    print("\nModel Configuration:")
    print(json.dumps(model_config, indent=2))

if __name__ == "__main__":
    asyncio.run(main())

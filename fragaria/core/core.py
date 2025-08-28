"""Core module for Fragaria - Chain of Thought Reasoning Library"""

import os
import asyncio
import aiohttp
import openai
from typing import List, Dict, Optional, Any
import json
from collections import defaultdict
import random
import sqlite3
import time
import yaml
import math

# Load configuration
def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path is None:
        # Try to load from package directory first, then from current working directory
        package_config = os.path.join(os.path.dirname(__file__), "config.yaml")
        cwd_config = os.path.join(os.getcwd(), "config.yaml")
        
        # Prefer package config if it exists, otherwise use cwd config
        if os.path.exists(package_config):
            config_path = package_config
        elif os.path.exists(cwd_config):
            config_path = cwd_config
        else:
            config_path = package_config  # Default to package path for error message
            
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}. "
                                "Please create a config.yaml file with your API keys.")
    
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)

class FragariaCore:
    def __init__(self, config_path: str = None):
        """Initialize FragariaCore with configuration"""
        self.config = load_config(config_path)
        self.api_key = None
        
        # Set up the OpenAI client based on the selected provider
        self.LLM_PROVIDER = self.config["llm_provider"]
        if self.LLM_PROVIDER == "openai":
            self.api_key = self.config["openai_api_key"]
            if self.api_key == "YOUR_OPENAI_API_KEY_HERE" or not self.api_key:
                raise ValueError("OpenAI API key not configured. Please update your config.yaml file with a valid API key.")
            self.client = openai.OpenAI(api_key=self.config["openai_api_key"])
        elif self.LLM_PROVIDER == "groq":
            self.api_key = self.config["groq_api_key"]
            if self.api_key == "YOUR_GROQ_API_KEY_HERE" or not self.api_key:
                raise ValueError("Groq API key not configured. Please update your config.yaml file with a valid API key.")
            self.client = openai.OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.config["groq_api_key"]
            )
        elif self.LLM_PROVIDER == "together":
            self.api_key = self.config["together_api_key"]
            if self.api_key == "YOUR_TOGETHER_API_KEY_HERE" or not self.api_key:
                raise ValueError("Together API key not configured. Please update your config.yaml file with a valid API key.")
            self.client = openai.OpenAI(
                api_key=self.config["together_api_key"],
                base_url="https://api.together.xyz/v1",
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.LLM_PROVIDER}")
        
        # Model configuration
        self.model_config = self.config["model_config"][self.LLM_PROVIDER]
        
        # Database setup
        self.init_db()
    
    def init_db(self):
        """Initialize the database for storing CoT paths"""
        db_path = self.config["database"]["path"]
        # If path is relative, make it relative to the package directory
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.path.dirname(__file__), db_path)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS cot_paths
                     (problem_type TEXT, method TEXT, score REAL, uses INTEGER)''')
        conn.commit()
        conn.close()
    
    def _get_db_path(self) -> str:
        """Get the absolute path to the database"""
        db_path = self.config["database"]["path"]
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.path.dirname(__file__), db_path)
        return db_path
    
    async def call_openai_api(self, model: str, system_prompt: str, user_prompt: str) -> str:
        """Call the OpenAI API with the specified model and prompts"""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content.strip()
    
    async def classify_or_create_problem_type(self, text: str) -> str:
        """Classify the problem type or create a new one if needed"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT DISTINCT problem_type FROM cot_paths")
        known_problem_types = set(row[0] for row in c.fetchall())
        conn.close()
        
        system_prompt = "You are an AI assistant specialized in classifying problem types. Your task is to either classify the given text into an existing problem type or create a new suitable problem type if none of the existing ones fit well. You will respond in JSON."
        
        example_json = json.dumps({"problem_type": "<problem type>"})
        
        if not known_problem_types:
            user_prompt = (f"Create a suitable problem type classification for the following text. "
                          f"Respond with only the problem type name as a JSON. Example json is {example_json} "
                          f"\n\nText: {text}")
        else:
            types_str = ", ".join(known_problem_types)
            user_prompt = (f"Classify the following text into one of these problem types: {types_str}. "
                          f"If none fit well, create a new suitable problem type. "
                          f"Respond with only the problem type name as a JSON. Example json is {example_json} "
                          f"\n\nText: {text}")
        
        problem_type = await self.call_openai_api(self.model_config["classify"], system_prompt, user_prompt)
        problem_type = json.loads(problem_type)["problem_type"]
        return problem_type
    
    async def generate_cot_paths(self, text: str, problem_type: str) -> List[Dict[str, any]]:
        """Generate chain of thought paths for a given problem"""
        example_json = json.dumps({"approaches": [{"method":"<method name>","description":"<method description>","steps":["<detailed step 1>","<detailed step 2>","<detailed step 3>"]}]})
        system_prompt = "You are an AI assistant specialized in generating diverse chain of thought approaches for problem-solving. You will respond in JSON."
        user_prompt = (f"Generate a list of 3 different chain of thought approaches to analyze the following {problem_type} problem. "
                      f"Respond in JSON. Example {example_json} \n\nProblem: {text}")
        
        response = await self.call_openai_api(self.model_config["generate"], system_prompt, user_prompt)
        response = json.loads(response)
        return response["approaches"]
    
    async def run_cot_path(self, session: aiohttp.ClientSession, text: str, path: Dict[str, any], problem_type: str, system_prompt: str) -> Dict:
        """Run a single chain of thought path"""
        full_system_prompt = (f"{system_prompt}\n"
                             f"You are an AI assistant specialized in analyzing {problem_type} problems using specific chain of thought approaches. "
                             f"Your task is to apply the given approach to analyze the problem.")
        user_prompt = (f"Analyze the following problem using this chain of thought approach: {json.dumps(path)}"
                      f"\n\nProblem: {text}")
        
        # Use the correct API endpoint based on provider
        if self.LLM_PROVIDER == "openai":
            url = "https://api.openai.com/v1/chat/completions"
        elif self.LLM_PROVIDER == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
        elif self.LLM_PROVIDER == "together":
            url = "https://api.together.xyz/v1/chat/completions"
        
        async with session.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_config["analyze"],
                "messages": [
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
        ) as response:
            result = await response.json()
            return {"method": path, "result": result['choices'][0]['message']['content']}
    
    async def combine_results(self, results: List[Dict], problem_type: str, system_prompt: str) -> str:
        """Combine results from multiple chain of thought paths"""
        example_json = {"results": {"<method name>": "<text paragraph>"}}
        full_system_prompt = (f"{system_prompt}\n"
                             f"You are an AI assistant specialized in combining and summarizing multiple analysis results for {problem_type} problems. "
                             f"Your task is to synthesize the given results into a coherent summary. You will respond in JSON.")
        results_text = "\n\n".join([f"Method: {json.dumps(r['method'])}\nResult: {r['result']}" for r in results])
        user_prompt = (f"Combine and summarize the following analysis results as a JSON with a results key mapped to a text paragraph per result. "
                      f"Do not modify the method name. Example {json.dumps(example_json)} \n\n{results_text}")
        
        final_result = await self.call_openai_api(self.model_config["combine"], full_system_prompt, user_prompt)
        final_result = json.loads(final_result)
        return final_result
    
    async def evaluate_result(self, text: str, result: str, problem_type: str, system_prompt: str) -> Dict[str, float]:
        """Evaluate the quality of the results"""
        example_json = {"<method 1>": "<score>", "<method 2>": "<score>", "<method 3>": "<score>"}
        full_system_prompt = (f"{system_prompt}\n"
                             f"You are an AI assistant specialized in evaluating the quality of analysis for {problem_type} problems. "
                             f"Your task is to rate the given result on a scale from 0 to 10, where 10 is the highest quality.")
        user_prompt = (f"On a scale of 0 to 10, how well does this result answer or analyze the given problem? "
                      f"Provide only the numerical score in the JSON format for every type of result. "
                      f"Please retain the method name. Example {json.dumps(example_json)} "
                      f"\n\nProblem: {text}\n\nResult: {json.dumps(result)}")
        
        scores = await self.call_openai_api(self.model_config["evaluate"], full_system_prompt, user_prompt)
        scores = json.loads(scores)
        return scores
    
    async def update_cot_scores(self, problem_type: str, paths: List[Dict[str, any]], scores: Dict[str, float]):
        """Update the scores for chain of thought paths in the database"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(db_path)
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
    
    def select_cot_paths(self, problem_type: str, n: int = 3) -> List[Dict[str, any]]:
        """Select chain of thought paths using UCB algorithm"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(db_path)
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
    
    async def adapt_cot_path(self, path: Dict[str, any], problem_type: str, text: str, system_prompt: str) -> Dict[str, any]:
        """Adapt a chain of thought path for a new problem"""
        full_system_prompt = (f"{system_prompt}\n"
                             f"You are an AI assistant specialized in adapting chain of thought approaches for new problems. "
                             f"Your task is to modify the given approach to better suit the current problem while maintaining its core strategy.")
        user_prompt = (f"Adapt the following chain of thought approach for the current {problem_type} problem:"
                      f"\n\nOriginal approach: {json.dumps(path)}"
                      f"\n\nCurrent problem: {text}"
                      f"\n\nProvide the adapted approach in the same JSON format as the original.")
        
        adapted_path = await self.call_openai_api(self.model_config["adapt"], full_system_prompt, user_prompt)
        adapted_path = json.loads(adapted_path)
        return adapted_path
    
    async def parallel_cot_reasoning(self, text: str, system_prompt: str = "") -> Dict[str, any]:
        """Perform parallel chain of thought reasoning on a problem"""
        problem_type = await self.classify_or_create_problem_type(text)
        
        stored_paths = self.select_cot_paths(problem_type)
        cot_paths = []
        for path in stored_paths:
            if path["uses"] == 0:
                # For new paths, generate a new approach
                new_path = await self.generate_cot_paths(text, problem_type)
                cot_paths.extend(new_path)
            else:
                # For existing paths, adapt them
                adapted_path = await self.adapt_cot_path(path, problem_type, text, system_prompt)
                cot_paths.append(adapted_path)
        
        async with aiohttp.ClientSession() as session:
            tasks = [self.run_cot_path(session, text, path, problem_type, system_prompt) for path in cot_paths]
            results = await asyncio.gather(*tasks)
            
        final_result = await self.combine_results(results, problem_type, system_prompt)
        scores = await self.evaluate_result(text, final_result, problem_type, system_prompt)
        highest_score_method = max(scores, key=scores.get)
        
        await self.update_cot_scores(problem_type, cot_paths, scores)
        
        return {
            "result": final_result["results"][highest_score_method],
            "all_results": final_result["results"],
            "scores": scores,
            "problem_type": problem_type
        }

# Convenience function for simple usage
async def analyze_problem(text: str, system_prompt: str = "", config_path: str = None) -> Dict[str, any]:
    """Analyze a problem using Fragaria's chain of thought reasoning"""
    core = FragariaCore(config_path)
    return await core.parallel_cot_reasoning(text, system_prompt)